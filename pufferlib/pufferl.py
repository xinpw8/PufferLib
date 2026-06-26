## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full details
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import sys
import glob
import json
import ast
import time
import argparse
import configparser
import random
from collections import defaultdict
import multiprocessing as mp
from copy import deepcopy

import numpy as np

import torch
import pufferlib
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import PufferLib C++ backend. If you have non-default PyTorch, try installing with --no-build-isolation')

from pufferlib import selfplay
from pufferlib import league

import rich
import rich.traceback
from rich.table import Table
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v

def abbreviate(num, b2, c2):
    prefixes = ['', 'K', 'M', 'B', 'T']
    for i, prefix in enumerate(prefixes):
        if num < 1e3: break
        num /= 1e3

    return f'{b2}{num:.1f}{c2}{prefix}'

def duration(seconds, b2, c2):
    if seconds < 0: return f"{b2}0{c2}s"
    if seconds < 1: return f"{b2}{seconds*1000:.0f}{c2}ms"
    seconds = int(seconds)
    d = f'{b2}{seconds // 86400}{c2}d '
    h = f'{b2}{(seconds // 3600) % 24}{c2}h '
    m = f'{b2}{(seconds // 60) % 60}{c2}m '
    s = f'{b2}{seconds % 60}{c2}s'
    return d + h + m + s

def fmt_perf(name, color, delta_ref, elapsed, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*elapsed/delta_ref - 1e-5)
    return f'{color}{name}', duration(elapsed, b2, c2), f'{b2}{percent:2d}{c2}%'

def print_dashboard(args, model_size, flat_logs, clear=False, idx=[0],
        c1='[cyan]', c2='[white]', b1='[bright_cyan]', b2='[bright_white]'):
    g = lambda k, d=0: flat_logs.get(k, d)
    console = rich.console.Console()
    dashboard = Table(box=rich.box.ROUNDED, expand=True,
        show_header=False, border_style='bright_cyan')
    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)

    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=18)
    table.add_column(justify="right", width=12)

    table.add_row(
        f'{b1}PufferLib {b2}4.0 {idx[0]*" "}:blowfish:',
        f'{c1}GPU: {b2}{g("util/gpu_percent"):.0f}{c2}%',
        f'{c1}VRAM: {b2}{g("util/vram_used_gb"):.1f}{c2}/{b2}{g("util/vram_total_gb"):.0f}{c2}G',
        f'{c1}RAM: {b2}{g("util/cpu_mem_gb"):.1f}{c2}G',
    )
    idx[0] = (idx[0] - 1) % 10

    s = Table(box=None, expand=True)
    remaining = f'{b2}A hair past a freckle{c2}'
    agent_steps = g('agent_steps')
    if g('SPS') != 0:
        remaining = duration((args['train']['total_timesteps']*args['train'].get('gpus', 1) - agent_steps)/g('SPS'), b2, c2)

    s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
    s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
    s.add_row(f'{c2}Env', f'{b2}{args["env_name"]}')
    s.add_row(f'{c2}Params', abbreviate(model_size, b2, c2))
    s.add_row(f'{c2}Steps', abbreviate(agent_steps, b2, c2))
    s.add_row(f'{c2}SPS', abbreviate(g('SPS'), b2, c2))
    s.add_row(f'{c2}Epoch', f'{b2}{g("epoch")}')
    s.add_row(f'{c2}Uptime', duration(g('uptime'), b2, c2))
    s.add_row(f'{c2}Remaining', remaining)

    rollout = g('perf/rollout')
    train = g('perf/train')
    delta = rollout + train
    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf('Evaluate', b1, delta, rollout, b2, c2))
    p.add_row(*fmt_perf('  GPU', b2, delta, g('perf/eval_gpu'), b2, c2))
    p.add_row(*fmt_perf('  Env', b2, delta, g('perf/eval_env'), b2, c2))
    p.add_row(*fmt_perf('Train', b1, delta, train, b2, c2))
    p.add_row(*fmt_perf('  Misc', b2, delta, g('perf/train_misc'), b2, c2))
    p.add_row(*fmt_perf('  Forward', b2, delta, g('perf/train_forward'), b2, c2))

    l = Table(box=None, expand=True)
    l.add_column(f'{c1}Losses', justify="left", width=16)
    l.add_column(f'{c1}Value', justify="right", width=8)
    for k, v in flat_logs.items():
        if k.startswith('loss/'):
            l.add_row(f'{b2}{k[5:]}', f'{b2}{v:.3f}')

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, l)
    dashboard.add_row(monitor)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    left = Table(box=None, expand=True)
    right = Table(box=None, expand=True)
    table.add_row(left, right)
    left.add_column(f"{c1}User Stats", justify="left", width=20)
    left.add_column(f"{c1}Value", justify="right", width=10)
    right.add_column(f"{c1}User Stats", justify="left", width=20)
    right.add_column(f"{c1}Value", justify="right", width=10)

    i = 0
    for k, v in flat_logs.items():
        if k.startswith('env/') and k != 'env/n':
            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{k[4:]}', f'{b2}{v:.3f}')
            i += 1
            if i == 30:
                break

    if clear:
        console.clear()

    with console.capture() as capture:
        console.print(dashboard)

    print('\033[0;0H' + capture.get())

def validate_config(args):
    minibatch_size = args['train']['minibatch_size']
    horizon = args['train']['horizon']
    total_agents = args['vec']['total_agents']
    assert (minibatch_size % horizon) == 0, \
        f'minibatch_size {minibatch_size} must be divisible by horizon {horizon}'
    assert minibatch_size <= horizon * total_agents, \
        f'minibatch_size {minibatch_size} > total_agents {total_agents} * horizon {horizon}'

def _resolve_backend(args):
    compiled_env = getattr(_C, 'env_name', None)
    assert compiled_env is None or compiled_env == args['env_name'], \
        f'build.sh was run for {compiled_env}, not {args["env_name"]}'
    if args.get('slowly'):
        from pufferlib.torch_pufferl import PuffeRL
        return PuffeRL
    return _C

def _train_worker(args):
    backend = _resolve_backend(args)
    pufferl = backend.create_pufferl(args)
    args.pop('nccl_id', None)
    while pufferl.global_step < args['train']['total_timesteps']:
        backend.rollouts(pufferl)
        backend.train(pufferl)

    backend.close(pufferl)

def _train(env_name, args, sweep_obj=None, result_queue=None, verbose=False):
    '''Single-GPU training worker. Process target for both DDP ranks and sweep trials.'''
    backend = _resolve_backend(args)
    rank = args['rank']
    artifact_owner = rank == 0
    run_id = args.get('run_id') or str(int(1000*time.time()))
    if not bool(args.get('selfplay', {}).get('enabled', 0)):
        # Stale frozen-bank config from selfplay/match experiments should not
        # affect ordinary training. Otherwise uninitialized frozen policies own
        # part of the rollout rows and their episodes leak into env/* metrics.
        args['vec']['num_frozen_banks'] = 0
        args['vec']['frozen_bank_pct'] = 0.0

    if args['wandb'] and artifact_owner:
        import wandb
        wandb.init(id=run_id, config=args,
            project=args['wandb_project'], group=args['wandb_group'],
            tags=[args['tag']] if args['tag'] is not None else [],
            settings=wandb.Settings(console="off"),
        )

    target_key = f'env/{args["sweep"]["metric"]}'
    total_timesteps = args['train']['total_timesteps']
    all_logs = []

    # When sweeping, optionally score each trial with a final evaluator instead
    # of the training-time metric. Scripted-bot eval supersedes the older
    # fixed-checkpoint match mode so trials do not pay for both. League sweeps
    # are scored asynchronously by the shared match worker.
    league_mode = bool(args.get('sweep', {}).get('league', False))
    bot_eval_mode = bool(args.get('sweep', {}).get('bot_eval', False)) and not league_mode
    match_mode = (sweep_obj is not None
        and bool(args.get('sweep', {}).get('match_enemy_model_path'))
        and not bot_eval_mode
        and not league_mode)
    final_eval_mode = bot_eval_mode or match_mode
    final_checkpoint_mode = final_eval_mode or league_mode

    checkpoint_dir = os.path.join(args['checkpoint_dir'], args['env_name'], run_id)
    if artifact_owner:
        os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(args['log_dir'], args['env_name'])
    if artifact_owner:
        os.makedirs(log_dir, exist_ok=True)

    try:
        pufferl = backend.create_pufferl(args)
    except RuntimeError as e:
        print(f'WARNING: {e}, skipping')
        if artifact_owner and result_queue is not None:
            result_queue.put((args['gpu_id'], [], [], []))
        return

    args.pop('nccl_id', None)
    model_size = pufferl.num_params()
    if verbose:
        flat_logs = dict(unroll_nested_dict(backend.log(pufferl)))
        print_dashboard(args, model_size, flat_logs, clear=True)

    # Selfplay-pool curriculum (no-op unless selfplay.enabled). Disabled
    # under match-mode sweeps since match() owns its own perm/frozen bank.
    pool_state = None
    try:
        pool_state = selfplay.setup(
            pufferl, backend, args, run_id, artifact_owner=artifact_owner)
    except RuntimeError as e:
        print(f'WARNING: {e}, skipping')
        backend.close(pufferl)
        if artifact_owner and result_queue is not None:
            result_queue.put((args['gpu_id'], [], [], []))
        return

    model_path = ''
    flat_logs = {}
    train_epochs = int(total_timesteps // (args['vec']['total_agents'] * args['train']['horizon']))
    eval_epochs = 0 if league_mode else train_epochs // 2
    max_trial_seconds = float(args.get('sweep', {}).get('max_trial_seconds', 0) or 0)
    trial_start_time = time.time()
    for epoch in range(train_epochs + eval_epochs):
        if epoch < train_epochs:
            selfplay.sync(pufferl, backend, pool_state)
        backend.rollouts(pufferl)

        if epoch < train_epochs:
            backend.train(pufferl)

        time_capped = (
            sweep_obj is not None
            and epoch < train_epochs
            and max_trial_seconds > 0
            and time.time() - trial_start_time >= max_trial_seconds
        )

        # In match-sweep mode we need the final checkpoint to feed into match().
        is_final = epoch == train_epochs - 1
        should_save = (epoch < train_epochs) and (
            (sweep_obj is None
                and (epoch % args['checkpoint_interval'] == 0 or is_final))
            or (final_checkpoint_mode and is_final)
        )
        if should_save and artifact_owner:
            model_path = os.path.join(checkpoint_dir, f'{pufferl.global_step:016d}.bin')
            backend.save_weights(pufferl, model_path)

        # Rate limit, but always log for eval and time caps to maintain determinism
        if (not time_capped
                and time.time() < pufferl.last_log_time + 0.6
                and epoch < train_epochs - 1):
            continue

        logs = backend.eval_log(pufferl) if epoch >= train_epochs else backend.log(pufferl)
        flat_logs = {**flat_logs, **dict(unroll_nested_dict(logs))}
        if time_capped:
            flat_logs['sweep/trial_time_capped'] = 1.0

        if epoch < train_epochs:
            selfplay.step(pufferl, backend, pool_state, flat_logs, epoch)

        if verbose:
            print_dashboard(args, model_size, flat_logs)

        if target_key not in flat_logs and not final_eval_mode and not league_mode:
            continue

        if args['wandb'] and artifact_owner:
            wandb.log(flat_logs, step=flat_logs['agent_steps'])

        if epoch < train_epochs:
            all_logs.append(flat_logs)

            if time_capped:
                break

            if (sweep_obj is not None
                    and not final_eval_mode
                    and not league_mode
                    and pufferl.global_step > min(0.20*total_timesteps, 100_000_000) and
                    sweep_obj.early_stop(flat_logs, target_key)):
                break
        elif flat_logs['env/n'] > args['eval_episodes']:
            break


    if artifact_owner:
        print_dashboard(args, model_size, flat_logs)
    # Final-score trials may have early-stopped before the in-loop save fired;
    # ensure we always have a checkpoint to feed the post-training evaluator.
    if final_checkpoint_mode and artifact_owner and not model_path:
        model_path = os.path.join(checkpoint_dir, f'{pufferl.global_step:016d}.bin')
        backend.save_weights(pufferl, model_path)

    if league_mode and artifact_owner:
        if not all_logs:
            all_logs.append(flat_logs)
        metrics = {k: [v] for k, v in all_logs[-1].items()}
        log_dir = os.path.join(args['log_dir'], args['env_name'])
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, run_id + '.json'), 'w') as f:
            json.dump({**args, 'metrics': metrics}, f)
        if args['wandb']:
            wandb.run.finish()
        if result_queue is not None:
            result_queue.put({
                'gpu_id': args['gpu_id'],
                'ok': bool(model_path),
                'run_id': run_id,
                'checkpoint_path': model_path,
                'hypers': deepcopy(args),
                'cost': float(metrics.get('uptime', [0.0])[-1]),
                'timesteps': int(metrics.get('agent_steps', [0])[-1]),
            })
        else:
            backend.close(pufferl)
        return

    backend.close(pufferl)

    if target_key not in flat_logs and not final_eval_mode and not league_mode:
        if artifact_owner and result_queue is not None:
            result_queue.put((args['gpu_id'], None, None, None))
        return

    if not artifact_owner:
        return

    # Match-mode scoring: primary = trained policy (model_path); frozen bank =
    # fixed enemy. Score is slot 0's average winrate. Creates its own pufferl
    # so must run after the training instance is closed. Single observation per
    # trial (mid-training curve doesn't predict final match score).
    match_score = None
    bot_eval_logs = None
    bot_perf = None
    if match_mode and artifact_owner:
        sweep_cfg = args['sweep']
        match_args = deepcopy(args)
        match_args['enemy_hidden_size'] = int(sweep_cfg['match_enemy_hidden_size'])
        match_args['enemy_num_layers'] = int(sweep_cfg['match_enemy_num_layers'])
        match_logs = match(env_name,
            policy_a_path=model_path,
            policy_b_path=sweep_cfg['match_enemy_model_path'],
            num_games=int(sweep_cfg['match_num_games']),
            args=match_args, verbose=verbose)
        match_score = float(match_logs['env/slot_0_score'])
        if args['wandb'] and artifact_owner:
            wandb.log({'env/match_score': match_score}, step=flat_logs['agent_steps'])

    if bot_eval_mode and artifact_owner:
        sweep_cfg = args['sweep']
        bot_eval_logs = eval_bot(env_name,
            policy_path=model_path,
            num_games=int(sweep_cfg['bot_eval_episodes']),
            eval_agents=int(sweep_cfg.get('bot_eval_envs', 0)),
            burnin_games=int(sweep_cfg.get('bot_eval_burnin_episodes', 0)),
            bot_policy=int(sweep_cfg['bot_eval_policy']),
            max_ticks=int(sweep_cfg['bot_eval_max_ticks']),
            args=deepcopy(args), verbose=verbose)
        bot_perf = float(bot_eval_logs['env/perf'])
        if args['wandb'] and artifact_owner:
            wandb.log({
                'env/bot_perf': bot_perf,
                'env/bot_score': float(bot_eval_logs.get('env/score', 0.0)),
                'env/bot_damage_received': float(bot_eval_logs.get('env/damage_received', 0.0)),
                'env/bot_slot_0_score': float(bot_eval_logs.get('env/slot_0_score', 0.0)),
                'env/bot_draw_rate': float(bot_eval_logs.get('env/draw_rate', 0.0)),
            }, step=flat_logs['agent_steps'])

    # This version has the training perf logs and eval env logs
    all_logs.append(flat_logs)

    # Downsample results. Log keys can appear late, e.g. env/perf only after
    # eval epochs. For downsample=1, keep exactly the final point.
    n = args['sweep']['downsample']
    if n <= 1:
        metrics = {k: [v] for k, v in all_logs[-1].items()}
    else:
        def _reduce(values):
            if not values:
                return None
            try:
                return float(np.mean(values))
            except (TypeError, ValueError):
                return values[-1]

        metrics = {k: [[]] for k in all_logs[0]}
        logged_timesteps = all_logs[-1]['agent_steps']
        next_bin = logged_timesteps / (n - 1)
        for log in all_logs:
            for k, v in log.items():
                if k not in metrics:
                    prior_bins = max(len(metrics['agent_steps']) - 1, 0)
                    metrics[k] = [v] * prior_bins + [[]]
                metrics[k][-1].append(v)

            if log['agent_steps'] < next_bin:
                continue

            next_bin += logged_timesteps / (n - 1)
            for k in list(metrics):
                reduced = _reduce(metrics[k][-1])
                if reduced is None and len(metrics[k]) > 1:
                    reduced = metrics[k][-2]
                metrics[k][-1] = reduced
                metrics[k].append([])

        for k in list(metrics):
            if k in all_logs[-1]:
                metrics[k][-1] = all_logs[-1][k]
            else:
                reduced = _reduce(metrics[k][-1])
                if reduced is None and len(metrics[k]) > 1:
                    reduced = metrics[k][-2]
                metrics[k][-1] = reduced

    # Match-mode: single observation at final-training cost. Protein's curve
    # fit collapses to one point — we only trust the match winrate, not any
    # training-time proxy. Replicate the scalar across all downsample bins so
    # the JSON log shape matches every other metric (cache_data.py rejects
    # length-mismatched metrics as "bad data").
    if match_mode:
        metrics['env/match_score'] = [match_score] * len(metrics['agent_steps'])
    if bot_eval_mode and bot_eval_logs is not None:
        metrics['env/bot_perf'] = [bot_perf] * len(metrics['agent_steps'])
        for src_key, dst_key in (
                ('env/score', 'env/bot_score'),
                ('env/damage_received', 'env/bot_damage_received'),
                ('env/episode_length', 'env/bot_episode_length'),
                ('env/slot_0_score', 'env/bot_slot_0_score'),
                ('env/slot_1_score', 'env/bot_slot_1_score'),
                ('env/draw_rate', 'env/bot_draw_rate'),
                ('env/n', 'env/bot_n')):
            if src_key in bot_eval_logs:
                metrics[dst_key] = [float(bot_eval_logs[src_key])] * len(metrics['agent_steps'])

    # Save own log: config + downsampled results
    if artifact_owner:
        log_dir = os.path.join(args['log_dir'], args['env_name'])
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, run_id + '.json'), 'w') as f:
            json.dump({**args, 'metrics': metrics}, f)

    if args['wandb'] and artifact_owner:
        if sweep_obj is None and model_path: # Don't spam uploads during sweeps
            artifact = wandb.Artifact(run_id, type='model')
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)

        wandb.run.finish()

    if artifact_owner and result_queue is not None:
        if league_mode:
            result_queue.put({
                'gpu_id': args['gpu_id'],
                'ok': bool(model_path),
                'run_id': run_id,
                'checkpoint_path': model_path,
                'hypers': deepcopy(args),
                'cost': float(metrics.get('uptime', [0.0])[-1]),
                'timesteps': int(metrics.get('agent_steps', [0])[-1]),
            })
        elif bot_eval_mode and bot_perf is not None:
            # One observation: final hypers -> scripted-bot perf, at total training cost.
            result_queue.put((args['gpu_id'], [bot_perf],
                [metrics['uptime'][-1]], [metrics['agent_steps'][-1]]))
        elif match_mode:
            # One observation: final hypers -> match winrate, at total training cost.
            result_queue.put((args['gpu_id'], [match_score],
                [metrics['uptime'][-1]], [metrics['agent_steps'][-1]]))
        else:
            result_queue.put((args['gpu_id'], metrics[target_key], metrics['uptime'], metrics['agent_steps']))


def train(env_name, args=None, gpus=None, **kwargs):
    args = args or load_config(env_name)
    validate_config(args)

    subprocess = gpus is not None
    gpus = list(gpus or range(args['train']['gpus']))
    args['world_size'] = len(gpus)
    if not args.get('run_id'):
        if args.get('wandb'):
            import wandb
            args['run_id'] = wandb.util.generate_id()
        else:
            args['run_id'] = str(int(1000*time.time()))
    args['nccl_id'] = _C.get_nccl_id() if len(gpus) > 1 else b''

    if not subprocess:
        gpus = gpus[-1:] + gpus[:-1]  # Main process gets rank 0

    ctx = mp.get_context('spawn')
    for rank, gpu_id in reversed(list(enumerate(gpus))):
        worker_args = deepcopy(args)
        worker_args['rank'] = rank
        worker_args['gpu_id'] = gpu_id
        if rank == 0 and not subprocess:
            _train(env_name, worker_args, verbose=True)
        else:
            # Protein's GP models live on cuda:0 on non-WSL setups; spawn-pickling
            # them works fine via CUDA IPC. On WSL, sweep.py forces device='cpu'
            # at construction so there's nothing to move.
            ctx.Process(target=_train, args=(env_name, worker_args),
                kwargs=kwargs).start()


def _league_arch(args):
    return {
        'hidden_size': int(float(args['policy']['hidden_size'])),
        'num_layers': int(float(args['policy']['num_layers'])),
    }


def _strip_league_arch_sweeps(sweep_config):
    # Historical-selfplay league trials do not load checkpoints from other
    # trials during training, so model size can be swept safely. Kept as a
    # compatibility shim for older call sites.
    return None


def _league_state_path(env_name, args):
    sweep_cfg = args['sweep']
    configured = sweep_cfg.get('league_state_path') or ''
    if configured:
        sweep_id = os.path.basename(configured)
        if sweep_id.endswith('_league.json'):
            sweep_id = sweep_id[:-len('_league.json')]
        else:
            sweep_id = os.path.splitext(sweep_id)[0]
    else:
        sweep_id = str(args.get('run_id') or int(1000*time.time()))
        configured = os.path.join(args['log_dir'], env_name, f'{sweep_id}_league.json')
        sweep_cfg['league_state_path'] = configured
    args['sweep_id'] = sweep_id
    return configured, sweep_id


def _validate_and_force_league_config(env_name, args, pareto=False):
    if pareto:
        raise ValueError('league mode does not support paretosweep')
    if env_name != 'robocode':
        raise ValueError('league sweep mode is currently implemented for robocode')
    args['train']['gpus'] = 1
    if not bool(args.get('selfplay', {}).get('enabled', 0)):
        raise ValueError('league sweep mode requires selfplay.enabled = 1')
    if int(args.get('env', {}).get('num_agents', 0)) != 2:
        raise ValueError('league sweep mode requires env.num_agents = 2')
    if int(args.get('env', {}).get('num_bots', 0)) != 0:
        raise ValueError('league sweep mode requires env.num_bots = 0')

    sweep_cfg = args['sweep']
    _strip_league_arch_sweeps(sweep_cfg)
    sweep_cfg['metric'] = 'elo'
    sweep_cfg['downsample'] = 1
    sweep_cfg['max_trial_seconds'] = 0
    sweep_cfg['bot_eval'] = False
    sweep_cfg['match_enemy_model_path'] = ''
    sweep_cfg['match_enemy_hidden_size'] = 0
    sweep_cfg['match_enemy_num_layers'] = 0

    if int(sweep_cfg.get('league_match_gpus', 1)) != 1:
        raise ValueError('league sweep mode currently supports exactly one match GPU')


def _configure_league_trial_args(args):
    args.setdefault('selfplay', {})['enabled'] = 1
    # League sweeps use the league only for post-hoc Elo scoring. Each trial
    # remains a reproducible ordinary historical-selfplay run.
    args['selfplay']['external_opponent_state_path'] = ''
    args['vec']['frozen_bank_hidden_size'] = int(float(args['policy']['hidden_size']))
    args['vec']['frozen_bank_num_layers'] = int(float(args['policy']['num_layers']))
    args.setdefault('env', {})['num_agents'] = 2
    args['env']['num_bots'] = 0


def _materialize_league_anchor(env_name, args, state_path, sweep_id, gpu_id):
    arch = _league_arch(args)
    state = league.read_state(state_path)
    if state is not None:
        for player in state.get('players', []):
            if player.get('id') == league.ANCHOR_ID and player.get('checkpoint_path'):
                if os.path.exists(player['checkpoint_path']):
                    return player['checkpoint_path']

    anchor_dir = os.path.join(args['checkpoint_dir'], env_name, f'{sweep_id}_league_anchor')
    os.makedirs(anchor_dir, exist_ok=True)
    anchor_path = os.path.join(anchor_dir,
        f'random_h{arch["hidden_size"]}_l{arch["num_layers"]}.bin')
    if not os.path.exists(anchor_path):
        cfg = deepcopy(args)
        cfg['reset_state'] = False
        cfg['rank'] = 0
        cfg['world_size'] = 1
        cfg['gpu_id'] = gpu_id
        cfg['nccl_id'] = b''
        cfg.setdefault('selfplay', {})['enabled'] = 0
        cfg['vec']['num_buffers'] = 1
        cfg['vec']['total_agents'] = max(128, int(cfg.get('env', {}).get('num_agents', 2)))
        cfg['vec']['num_frozen_banks'] = 0
        cfg['vec']['frozen_bank_pct'] = 0.0
        cfg.setdefault('env', {})['dr'] = 0.0
        cfg['env']['num_agents'] = 2
        cfg['env']['num_bots'] = 0
        cfg['train']['horizon'] = 1
        backend = _resolve_backend(cfg)
        if backend is not _C:
            raise RuntimeError('league random anchor creation requires the native CUDA backend')
        pufferl = backend.create_pufferl(cfg)
        backend.save_weights(pufferl, anchor_path)

    league.ensure_anchor(state_path, anchor_path, arch, hypers={'policy': deepcopy(args['policy'])})
    return anchor_path


def _refresh_league_observations(sweep_obj, state_path):
    state = league.read_state(state_path)
    if state is None or not hasattr(sweep_obj, 'refresh_observations_by_run_id'):
        return 0
    return sweep_obj.refresh_observations_by_run_id(league.run_id_scores(state))


ROBOCODE_REWARD_CONDITIONING_KEYS = (
    'reward_melee_damage_inflicted',
    'reward_damage_taken',
    'reward_range_damage_inflicted',
)


def _player_reward_conditioning(player):
    env_cfg = (player.get('hypers') or {}).get('env') or {}
    return {
        key: float(env_cfg.get(key, 0.0) or 0.0)
        for key in ROBOCODE_REWARD_CONDITIONING_KEYS
    }


def _apply_match_reward_conditioning(match_args, player_a, player_b):
    env_cfg = match_args.setdefault('env', {})
    for slot, player in ((0, player_a), (1, player_b)):
        for key, value in _player_reward_conditioning(player).items():
            env_cfg[f'{key}_slot_{slot}'] = value


def _player_policy_arch(player, fallback_args=None):
    fallback_policy = (fallback_args or {}).get('policy') or {}
    player_arch = player.get('arch') or {}
    player_policy = (player.get('hypers') or {}).get('policy') or {}
    hidden = player_arch.get('hidden_size', player_policy.get(
        'hidden_size', fallback_policy.get('hidden_size', 128)))
    layers = player_arch.get('num_layers', player_policy.get(
        'num_layers', fallback_policy.get('num_layers', 1)))
    return {
        'hidden_size': int(float(hidden)),
        'num_layers': int(float(layers)),
    }


def _apply_match_policy_arch(match_args, player_a, player_b):
    policy_cfg = match_args.setdefault('policy', {})
    vec_cfg = match_args.setdefault('vec', {})
    a_arch = _player_policy_arch(player_a, match_args)
    b_arch = _player_policy_arch(player_b, match_args)
    policy_cfg['hidden_size'] = a_arch['hidden_size']
    policy_cfg['num_layers'] = a_arch['num_layers']
    match_args['enemy_hidden_size'] = b_arch['hidden_size']
    match_args['enemy_num_layers'] = b_arch['num_layers']
    vec_cfg['frozen_bank_hidden_size'] = b_arch['hidden_size']
    vec_cfg['frozen_bank_num_layers'] = b_arch['num_layers']


def _league_match_once_child(env_name, player_a, player_b, games, args, result_queue):
    try:
        match_args = deepcopy(args)
        match_args['match_eval_agents'] = int(args['sweep'].get('league_match_eval_agents', 8192))
        match_args['skip_match_close'] = True
        _apply_match_policy_arch(match_args, player_a, player_b)
        _apply_match_reward_conditioning(match_args, player_a, player_b)
        logs = match(env_name, player_a['path'], player_b['path'],
            num_games=int(games), args=match_args, verbose=False)
        result_queue.put({
            'ok': True,
            'score': float(logs.get('env/slot_0_score', 0.0)),
            'draw': float(logs.get('env/draw_rate', 0.0)),
            'games': int(logs.get('env/n', games)),
        })
    except BaseException as e:
        result_queue.put({'ok': False, 'error': f'{type(e).__name__}: {e}'})


def _league_match_once(env_name, player_a, player_b, games, args):
    ctx = mp.get_context('spawn')
    result_queue = ctx.SimpleQueue()
    proc = ctx.Process(target=_league_match_once_child,
        args=(env_name, player_a, player_b, int(games), args, result_queue))
    proc.start()
    proc.join(timeout=min(600, max(30, int(games) * 4)))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        raise RuntimeError(f'league match orientation timed out: {player_a["id"]} vs {player_b["id"]}')
    if result_queue.empty():
        raise RuntimeError(f'league match orientation exited without result: {player_a["id"]} vs {player_b["id"]}, exit={proc.exitcode}')
    result = result_queue.get()
    if not result.get('ok'):
        raise RuntimeError(result.get('error', 'league match orientation failed'))
    return result['score'], result['draw'], result['games']


def _league_run_pair(env_name, player_a, player_b, games, args):
    games = int(games)
    if player_a['id'] == player_b['id']:
        return 0.5, 1.0, games
    if games < 2:
        return _league_match_once(env_name, player_a, player_b, games, args)

    games_ab = games // 2
    games_ba = games - games_ab
    score_ab, draw_ab, n_ab = _league_match_once(env_name, player_a, player_b, games_ab, args)
    score_ba, draw_ba, n_ba = _league_match_once(env_name, player_b, player_a, games_ba, args)
    total = max(n_ab + n_ba, 1)
    a_score = (score_ab * n_ab + (1.0 - score_ba) * n_ba) / total
    draw = (draw_ab * n_ab + draw_ba * n_ba) / total
    return a_score, draw, total


def _league_match_worker(env_name, args, state_path, gpu_id, stop_event):
    worker_args = deepcopy(args)
    worker_args['gpu_id'] = gpu_id
    rng = random.Random(int(args.get('seed', 0)) + 1009 * (gpu_id + 1))
    games = int(args['sweep'].get('league_match_games', 4096))
    anchor_prob = float(args['sweep'].get('league_anchor_prob', 0.12))
    while not stop_event.is_set():
        try:
            state = league.read_state(state_path)
            if state is None:
                if stop_event.wait(2.0):
                    break
                continue
            player_a, player_b = league.choose_match_pair(
                state, rng=rng, anchor_prob=anchor_prob)
            a_score, draw, total = _league_run_pair(
                env_name, player_a, player_b, games, worker_args)
            ratings = league.record_match(
                state_path, player_a['id'], player_b['id'], total, a_score, draw)
            print(
                f'league_match {player_a["id"]} vs {player_b["id"]} '
                f'games={total} a_score={a_score:.4f} draw={draw:.4f} '
                f'elo=({ratings.get(player_a["id"], 0.0):.1f}, '
                f'{ratings.get(player_b["id"], 0.0):.1f})'
            )
        except RuntimeError as e:
            print(f'league_match waiting: {e}')
            if stop_event.wait(1.0):
                break
        except Exception as e:
            print(f'WARNING: league match worker error: {e}')
            if stop_event.wait(15.0):
                break


def _league_sweep(env_name, args=None, pareto=False):
    args = args or load_config(env_name)
    sweep_gpus = args['sweep']['gpus'] or len(os.listdir('/proc/driver/nvidia/gpus'))
    _validate_and_force_league_config(env_name, args, pareto=pareto)
    exp_gpus = int(args['train']['gpus'])

    match_gpus = int(args['sweep'].get('league_match_gpus', 1))
    train_slots_cfg = int(args['sweep'].get('league_train_gpus', 0) or (sweep_gpus - match_gpus))
    if sweep_gpus <= match_gpus:
        raise ValueError('league sweep requires at least one training GPU and one match GPU')
    train_slots = min(train_slots_cfg, sweep_gpus - match_gpus)
    if train_slots < 1:
        raise ValueError('league sweep has no training GPU slots')

    all_gpu_ids = list(range(sweep_gpus))
    match_gpu_ids = all_gpu_ids[-match_gpus:]
    train_gpu_ids = [gpu for gpu in all_gpu_ids if gpu not in match_gpu_ids][:train_slots]
    args['no_model_upload'] = True

    state_path, sweep_id = _league_state_path(env_name, args)
    arch = _league_arch(args)
    state = league.load_or_create(state_path, sweep_id, arch=arch, config={
        'env_name': env_name,
        'league_match_games': int(args['sweep'].get('league_match_games', 4096)),
        'trial_opponents': 'historical_selfplay_only',
    })
    _materialize_league_anchor(env_name, args, state_path, sweep_id, match_gpu_ids[0])
    _configure_league_trial_args(args)

    sweep_config = args['sweep']
    method = sweep_config.pop('method')
    import pufferlib.sweep
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except Exception:
        raise ValueError(f'Invalid sweep method {method}. See pufferlib.sweep')
    sweep_obj = sweep_cls(sweep_config)
    num_experiments = int(args['sweep']['max_runs'])

    ctx = mp.get_context('spawn')
    result_queue = ctx.SimpleQueue()
    stop_event = ctx.Event()
    match_proc = ctx.Process(target=_league_match_worker,
        args=(env_name, deepcopy(args), state_path, match_gpu_ids[0], stop_event))
    match_proc.start()

    active = {}
    completed = 0
    launched = 0

    def collect_one():
        nonlocal completed
        result = result_queue.get()
        if isinstance(result, dict):
            gpu_id = result.get('gpu_id')
            done_args = active.pop(gpu_id)
            run_id = result.get('run_id') or done_args.get('run_id')
            if not result.get('ok'):
                sweep_obj.observe(done_args, 0, 0, is_failure=True, run_id=run_id)
                return

            timesteps = int(result.get('timesteps', done_args['train']['total_timesteps']))
            cost = float(result.get('cost', 0.0))
            done_args['train']['total_timesteps'] = timesteps
            player_hypers = result.get('hypers', done_args)
            player_arch = _league_arch(player_hypers)
            player = league.register_player(
                state_path, run_id, result['checkpoint_path'], player_hypers, cost, arch=player_arch)
            sweep_obj.observe(done_args, float(player.get('elo', 0.0)), cost,
                is_failure=False, run_id=run_id)
            _refresh_league_observations(sweep_obj, state_path)
            completed += 1
            return

        gpu_id, scores, costs, timesteps = result
        done_args = active.pop(gpu_id)
        sweep_obj.observe(done_args, 0, 0, is_failure=True, run_id=done_args.get('run_id'))

    try:
        while completed < num_experiments or active:
            if active and (len(active) >= train_slots or completed + len(active) >= num_experiments):
                collect_one()
                continue
            if completed + len(active) >= num_experiments:
                continue

            gpu_id = next(gpu for gpu in train_gpu_ids if gpu not in active)
            idx = completed + len(active)
            _refresh_league_observations(sweep_obj, state_path)
            if idx > 1:
                sweep_obj.suggest(args)
            _configure_league_trial_args(args)
            try:
                validate_config(args)
            except (AssertionError, ValueError) as e:
                print(f'WARNING: {e}, skipping')
                sweep_obj.observe(args, 0, 0, is_failure=True, run_id=None)
                continue

            exp_args = deepcopy(args)
            exp_args['run_id'] = f'{sweep_id}_{launched:05d}'
            exp_args['gpu_id'] = gpu_id
            exp_args['sweep']['league_state_path'] = state_path
            active[gpu_id] = exp_args
            launched += 1
            train(env_name, exp_args, range(gpu_id, gpu_id + exp_gpus),
                sweep_obj=sweep_obj, result_queue=result_queue)
    finally:
        shutdown_timeout = min(600, max(30, int(args['sweep'].get('league_match_games', 4096)) * 2))
        deadline = time.time() + shutdown_timeout
        while match_proc.is_alive() and time.time() < deadline:
            state = league.read_state(state_path)
            policies = [p for p in state.get('players', []) if p.get('kind') == 'policy'] if state else []
            if not policies or state.get('matches'):
                break
            time.sleep(1.0)

        stop_event.set()
        match_proc.join(timeout=shutdown_timeout)
        if match_proc.is_alive():
            match_proc.terminate()
            match_proc.join(timeout=5)

def sweep(env_name, args=None, pareto=False):
    '''Train entry point. Handles single-GPU, multi-GPU DDP, and sweeps.'''
    args = args or load_config(env_name)
    if bool(args.get('sweep', {}).get('league', False)):
        return _league_sweep(env_name, args=args, pareto=pareto)

    exp_gpus = args['train']['gpus']
    sweep_gpus = args['sweep']['gpus'] or len(os.listdir('/proc/driver/nvidia/gpus'))
    args['no_model_upload'] = True

    sweep_config = args['sweep']
    method = sweep_config.pop('method')
    import pufferlib.sweep
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise ValueError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep_obj = sweep_cls(sweep_config)
    num_experiments = args['sweep']['max_runs']
    ts_default = args['train']['total_timesteps']
    ts_config = sweep_config.get('train', {}).get('total_timesteps', {'min': ts_default, 'max': ts_default})
    
    all_timesteps = np.geomspace(ts_config['min'], ts_config['max'], sweep_gpus)
    result_queue = mp.get_context('spawn').Queue()

    active = {}
    completed = 0
    while completed < num_experiments:
        if len(active) >= sweep_gpus//exp_gpus: # Collect completed runs
            gpu_id, scores, costs, timesteps = result_queue.get()
            done_args = active.pop(gpu_id)

            if not scores:
                sweep_obj.observe(done_args, 0, 0, is_failure=True)
            else:
                completed += 1

            for s, c, t in zip(scores, costs, timesteps):
                done_args['train']['total_timesteps'] = t
                sweep_obj.observe(done_args, s, c, is_failure=False)

        idx = completed + len(active)
        if idx >= num_experiments:
            break # All experiments launched

        # TODO: only 1 per sweep etc
        gpu_id = next(i for i in range(sweep_gpus) if i not in active)
        timestep_total = all_timesteps[gpu_id] if pareto else None
        if idx > 1: # First experiment uses defaults
            sweep_obj.suggest(args, fixed_total_timesteps=timestep_total)

        try:
            validate_config(args)
        except (AssertionError, ValueError) as e:
            print(f'WARNING: {e}, skipping')
            sweep_obj.observe(args, 0, 0, is_failure=True)
            continue

        exp_args = deepcopy(args)
        active[gpu_id] = exp_args
        train(env_name, exp_args, range(gpu_id, gpu_id + exp_gpus),
            sweep_obj=sweep_obj, result_queue=result_queue)


def eval_bot(env_name, policy_path, num_games=4096, eval_agents=0, burnin_games=0,
        bot_policy=-1, max_ticks=0, args=None, verbose=True):
    '''Evaluate a trained policy against the env's scripted bot.'''
    args = args or load_config(env_name)
    args['reset_state'] = False
    args['train']['horizon'] = 1
    args['world_size'] = 1
    args['rank'] = 0
    args.setdefault('nccl_id', b'')

    num_games = int(num_games)
    burnin_games = int(burnin_games)
    eval_agents = int(eval_agents)
    if num_games <= 0:
        raise ValueError('num_games must be positive')
    if burnin_games < 0:
        raise ValueError('burnin_games must be non-negative')

    args['vec']['num_buffers'] = 2
    if eval_agents <= 0:
        # Avoid scoring only the first wave of completed episodes. If env count
        # ~= game count, quick wins finish first and slow losses are censored.
        eval_agents = min(4096, max(1024, num_games // 8))
        eval_agents = min(eval_agents, max(1024, num_games))
    else:
        # Explicit eval_agents means the caller is choosing the noise/runtime
        # tradeoff, e.g. tiny validation evals during sweeps.
        eval_agents = min(eval_agents, num_games)
    eval_agents += (-eval_agents) % args['vec']['num_buffers']
    args['vec']['total_agents'] = eval_agents
    args['vec']['num_frozen_banks'] = 0
    args['vec']['frozen_bank_pct'] = 0.0
    args.setdefault('selfplay', {})['enabled'] = 0
    args.setdefault('env', {})['dr'] = 0.0
    args['env']['num_agents'] = 1
    args['env']['num_bots'] = 1
    if bot_policy >= 0:
        args['env']['bot_policy'] = bot_policy
    if max_ticks > 0:
        args['env']['max_ticks'] = max_ticks

    backend = _resolve_backend(args)
    if backend is not _C:
        raise RuntimeError('eval_bot() requires the native CUDA backend')

    pufferl = backend.create_pufferl(args)
    backend.load_weights(pufferl, policy_path)

    def _delta_logs(current, baseline):
        if not baseline:
            return current
        n = float(current.get('env/n', 0.0))
        base_n = float(baseline.get('env/n', 0.0))
        delta_n = max(n - base_n, 0.0)
        if delta_n <= 0:
            return {'env/n': 0.0}
        out = {}
        for key, value in current.items():
            if key == 'env/n':
                out[key] = delta_n
            elif key.startswith('env/') and key in baseline:
                out[key] = (float(value) * n - float(baseline[key]) * base_n) / delta_n
            else:
                out[key] = value
        return out

    logs = {}
    baseline_logs = {}
    baseline_n = 0
    while True:
        backend.rollouts(pufferl)
        logs = dict(unroll_nested_dict(backend.eval_log(pufferl)))
        n = int(logs.get('env/n', 0))
        if burnin_games and not baseline_logs and n >= burnin_games:
            baseline_logs = logs.copy()
            baseline_n = n

        scored_logs = _delta_logs(logs, baseline_logs)
        scored_n = int(scored_logs.get('env/n', n))
        if verbose:
            perf = scored_logs.get('env/perf', 0.0)
            score = scored_logs.get('env/score', 0.0)
            if burnin_games and not baseline_logs:
                print(f'\rbot_eval_burnin={n}/{burnin_games}', end='')
            else:
                print(f'\rbot_eval={scored_n}/{num_games}  perf={perf:.4f}  score={score:.3f}', end='')
        if (n - baseline_n) >= num_games and (not burnin_games or baseline_logs):
            logs = scored_logs
            break

    if verbose:
        print()

    if not args.get('skip_match_close', False):
        backend.close(pufferl)
    return logs

def eval(env_name, args=None, load_path=None):
    '''Evaluate a trained policy. Supports both native and --slowly torch backends.'''
    args = args or load_config(env_name)
    args['reset_state'] = False
    args['train']['horizon'] = 1
    if 'env' in args and 'dr' in args['env']:
        args['env']['dr'] = 0.0

    backend = _resolve_backend(args)
    pufferl = backend.create_pufferl(args)

    # Resolve load path
    load_path = load_path or args.get('load_model_path')
    if load_path == 'latest':
        checkpoint_dir = args['checkpoint_dir']
        pattern = os.path.join(checkpoint_dir, args['env_name'], '**', '*.bin')
        candidates = glob.glob(pattern, recursive=True)
        if not candidates:
            raise FileNotFoundError(f'No .bin checkpoints found in {checkpoint_dir}/{args["env_name"]}/')
        load_path = max(candidates, key=os.path.getctime)

    if load_path is not None:
        backend.load_weights(pufferl, load_path)
        print(f'Loaded weights from {load_path}')

    #while True:
    for i in range(10000):
        #backend.render(pufferl, 0)
        backend.rollouts(pufferl)

    logs = dict(unroll_nested_dict(backend.eval_log(pufferl)))
    print('Perf: ', logs['env/perf'])
    backend.close(pufferl)

def match(env_name, policy_a_path, policy_b_path, num_games=4096, args=None, verbose=True):
    '''Head-to-head match between two trained policies in a 2-agent selfplay env.
    Policy A plays slot 0 (e.g. white in chess), policy B plays slot 1 (black).
    Both checkpoints must come from the same env / arch.
    '''
    args = args or load_config(env_name)
    args['reset_state'] = False
    args['train']['horizon'] = 1
    args.setdefault('nccl_id', b'')  # match is always single-GPU
    # Sweep suggestions can give odd agents_per_buffer (e.g. num_buffers=5,
    # total_agents=4096 -> 819). Pin to a stable eval config that guarantees
    # clean slot-0/slot-1 split; ignores trial's vec tuning (eval, not train).
    args['vec']['num_buffers'] = 2
    eval_agents = int(args.get('match_eval_agents')
        or args.get('sweep', {}).get('league_match_eval_agents', 8192)
        or 8192)
    args['vec']['total_agents'] = eval_agents
    args.setdefault('selfplay', {})['enabled'] = 0
    args.setdefault('env', {})['dr'] = 0.0
    args['env']['num_agents'] = 2
    args['env']['num_bots'] = 0
    backend = _resolve_backend(args)
    if backend is not _C:
        raise RuntimeError('match() requires the native CUDA backend')

    def _resolve_latest(path):
        if path != 'latest':
            return path
        pattern = os.path.join(args['checkpoint_dir'], args['env_name'], '**', '*.bin')
        candidates = glob.glob(pattern, recursive=True)
        if not candidates:
            raise FileNotFoundError(f'No .bin checkpoints found in {args["checkpoint_dir"]}/{args["env_name"]}/')
        return max(candidates, key=os.path.getctime)
    policy_a_path = _resolve_latest(policy_a_path)
    policy_b_path = _resolve_latest(policy_b_path)

    total_agents = int(args['vec']['total_agents'])
    num_buffers = int(args['vec']['num_buffers'])
    if total_agents % num_buffers != 0:
        raise RuntimeError(f'total_agents ({total_agents}) must be divisible by num_buffers ({num_buffers})')
    agents_per_buffer = total_agents // num_buffers
    half = agents_per_buffer // 2
    if 2 * half != agents_per_buffer:
        raise RuntimeError(f'agents_per_buffer ({agents_per_buffer}) must be even for 2-agent selfplay')

    # Primary holds policy A (owns first half of each buffer); one frozen bank
    # holds policy B (owns second half). Bank is created inside create_pufferl
    # before cudagraph capture so the graph bakes in its pointers; weight loads
    # later only update data.
    args['vec']['num_frozen_banks'] = 1
    args['vec']['frozen_bank_pct'] = 0.5
    # CLI flags take precedence; fall back to [sweep].match_enemy_* so the same
    # config drives sweep-time and CLI-time matches. 0 / None means "use primary".
    sweep_cfg = args.get('sweep', {})
    enemy_hidden = args.get('enemy_hidden_size') or sweep_cfg.get('match_enemy_hidden_size')
    enemy_layers = args.get('enemy_num_layers')  or sweep_cfg.get('match_enemy_num_layers')
    if enemy_hidden:
        args['vec']['frozen_bank_hidden_size'] = int(enemy_hidden)
    if enemy_layers:
        args['vec']['frozen_bank_num_layers'] = int(enemy_layers)

    pufferl = backend.create_pufferl(args)

    # Per-buffer perm: each env's slot 0 lands in primary's slice [0, half),
    # slot 1 lands in frozen bank's slice [half, agents_per_buffer). The env
    # side randomizes slot<->color per env, so A and B each play both colors.
    perm = np.empty(total_agents, dtype=np.int32)
    envs_per_buffer = half
    for b in range(num_buffers):
        off = b * agents_per_buffer
        for i in range(envs_per_buffer):
            perm[off + 2*i]     = off + i
            perm[off + 2*i + 1] = off + half + i
    backend.set_agent_perm(pufferl, perm)

    backend.load_weights(pufferl, policy_a_path)
    backend.load_frozen_bank(pufferl, 0, policy_b_path)

    logs = {}
    while True:
        backend.rollouts(pufferl)
        logs = dict(unroll_nested_dict(backend.eval_log(pufferl)))
        n = int(logs.get('env/n', 0))
        if verbose:
            a = logs.get('env/slot_0_score', 0.0)
            b = logs.get('env/slot_1_score', 0.0)
            draws = logs.get('env/draw_rate', 0.0)
            print(f'\rgames={n}/{num_games}  A={a:.3f}  B={b:.3f}  draw={draws:.3f}', end='')
        if n >= num_games:
            break

    if verbose:
        print()

    if not args.get('skip_match_close', False):
        backend.close(pufferl)
    return logs

def load_config(env_name):
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-enemy-model-path', type=str, default=None,
        help='Path to opponent checkpoint for `puffer match` (slot 1 / black in chess)')
    parser.add_argument('--num-games', type=int, default=4096,
        help='Number of games to play in `puffer match`')
    parser.add_argument('--enemy-hidden-size', type=int, default=None,
        help='hidden_size of the enemy checkpoint (defaults to primary)')
    parser.add_argument('--enemy-num-layers', type=int, default=None,
        help='num_layers of the enemy checkpoint (defaults to primary)')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandbrun')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='puffer4')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.add_argument('--slowly', action='store_true', help='Use PyTorch training backend')
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    puffer_config_dir = os.path.join(repo_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(repo_dir, 'config/default.ini')
    #CC: Remove the default. Just raise an error on "puffer train" etc with no env (think we already do)
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise ValueError('No config for env_name {}'.format(env_name))

    for section in p.sections():
        for key in p[section]:
            try:
                value = ast.literal_eval(p[section][key])
            except:
                value = p[section][key]

            #TODO: Can clean up with default sections in 3.13+
            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            dtype = type(value)
            parser.add_argument(
                fmt.replace('_', '-'), default=value,
                type=lambda v, t=dtype: v if v == 'auto' else t(v),
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        nxt = args
        for subkey in key.split('.'):
            prev = nxt
            nxt = nxt.setdefault(subkey, {})

        prev[subkey] = value

    args['env_name'] = env_name
    for section in p.sections():
        args.setdefault(section, {})
    return dict(args)

def main():
    err = 'Usage: puffer [train, eval, sweep, paretosweep, match] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise ValueError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    args = load_config(env_name)

    if 'train' in mode:
        train(env_name=env_name, args=args)
    elif 'eval' in mode:
        eval(env_name=env_name, args=args)
    elif 'sweep' in mode:
        sweep(env_name=env_name, args=args, pareto='pareto' in mode)
    elif 'match' in mode:
        a_path = args.get('load_model_path')
        b_path = args.get('load_enemy_model_path')
        if not a_path or not b_path:
            raise ValueError('puffer match requires --load-model-path and --load-enemy-model-path')
        match(env_name=env_name, policy_a_path=a_path, policy_b_path=b_path,
            num_games=args.get('num_games', 4096), args=args)
    else:
        raise ValueError(err)

if __name__ == '__main__':
    main()
