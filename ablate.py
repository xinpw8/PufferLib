import configparser
import argparse
import shutil
import glob
import uuid
import ast
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import clean_pufferl
   
def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args['train']['device'])

def init_wandb(args, name, id=None, resume=True, tag=None):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        group=args['wandb_group'],
        allow_val_change=True,
        save_code=False,
        resume=resume,
        config=args,
        name=name,
        tags=[tag] if tag is not None else [],
    )
    return wandb

def init_neptune(args, name, id=None, resume=True, tag=None):
    import neptune
    run = neptune.init_run(
        project="pufferai/ablations",
        capture_hardware_metrics=False,
        capture_stdout=False,
        capture_stderr=False,
        capture_traceback=False,
        tags=[tag] if tag is not None else [],
    )
    return run

import numpy as np

def log_normal(mean, scale, clip):
    '''Samples normally spaced points on a log 10 scale.
    mean: Your center sample point
    scale: standard deviation in base 10 orders of magnitude
    clip: maximum standard deviations

    Example: mean=0.001, scale=1, clip=2 will produce data from
    0.1 to 0.00001 with most of it between 0.01 and 0.0001
    '''
    return 10**np.clip(
        np.random.normal(
            np.log10(mean),
            scale,
        ),
        a_min = np.log10(mean) - clip,
        a_max = np.log10(mean) + clip,
    )

def logit_normal(mean, scale, clip):
    '''log normal but for logit data like gamma and gae_lambda'''
    return 1 - log_normal(1 - mean, scale, clip)

def uniform_pow2(min, max):
    '''Uniform distribution over powers of 2 between min and max inclusive'''
    min_base = np.log2(min)
    max_base = np.log2(max)
    return 2**np.random.randint(min_base, max_base+1)

def uniform(min, max):
    '''Uniform distribution between min and max inclusive'''
    return np.random.uniform(min, max)

def int_uniform(min, max):
    '''Uniform distribution between min and max inclusive'''
    return np.random.randint(min, max+1)


#samples = [log_normal(mu = 5e-3, scale = 1, clip = 2.0) for _ in range(7000)]
#samples = [logit_normal(mu = 0.98, scale = 0.5, clip = 1.0) for _ in range(7000)]
#samples = [uniform_pow2(min=2, max=64) for _ in range(7000)]

def sweep(args, env_name, make_env, policy_cls, rnn_cls):
    import random
    import numpy as np
    import time

    sweep_config = args['sweep']
    #for trial in range(10):
    while True:
        np.random.seed(int(time.time()))
        random.seed(int(time.time()))
        hypers = {}
        for group_key in ('train',):
            hypers[group_key] = {}
            param_group = sweep_config[group_key]
            for name, param in param_group.items():
                if 'values' in param:
                    assert 'distribution' not in param
                    choice = random.choice(param['values'])
                elif 'distribution' in param:
                    if param['distribution'] == 'uniform':
                        choice = random.uniform(param['min'], param['max'])
                    elif param['distribution'] == 'int_uniform':
                        choice = random.randint(param['min'], param['max']+1)
                    elif param['distribution'] == 'uniform_pow2':
                        choice = uniform_pow2(param['min'], param['max'])
                    elif param['distribution'] == 'log_normal':
                        choice = log_normal(
                            param['mean'], param['scale'], param['clip'])
                    elif param['distribution'] == 'logit_normal':
                        choice = logit_normal(
                            param['mean'], param['scale'], param['clip'])
                    else:
                        raise ValueError(f'Invalid distribution: {param["distribution"]}')
                else:
                    raise ValueError('Must specify either values or distribution')

                hypers[group_key][name] = choice

        if args['neptune']:
            run = init_neptune(args, env_name, id=args['exp_id'], tag=args['tag'])
            for k, v in pufferlib.utils.unroll_nested_dict(hypers):
                run[k].append(v)

            args['train'].update(hypers['train'])
            train(args, make_env, policy_cls, rnn_cls, neptune=run)
        elif args['wandb']:
            run = init_wandb(args, env_name, id=args['exp_id'], tag=args['tag'])
            train(args, make_env, policy_cls, rnn_cls, wandb=run)

### CARBS Sweeps
def convert_hyperparams(sweep_config):
    '''Convert our sweep config format to WandB's format
    This is required because WandB is too restrictive and doesn't
    let us specify extra data for custom algorithms like CARBS'''
    for group_key, param_group in sweep_config['parameters'].items():
        param_group = param_group['parameters']
        for name, param in param_group.items():
            is_integer = False
            if 'is_integer' in param:
                is_integer = param['is_integer']
                del param['is_integer']
            if 'scale' in param:
                del param['scale']
            if 'values' in param and 'distribution' in param:
                del param['distribution']
            if 'search_center' in param:
                del param['search_center']
            if 'distribution' in param:
                dist = param['distribution']
                overwrite = dist
                if dist == 'log':
                    overwrite = 'log_uniform_values'
                elif dist == 'linear':
                    if is_integer:
                        overwrite = 'int_uniform'
                    else:
                        overwrite = 'uniform'
                elif dist == 'logit':
                    overwrite = 'uniform'

                param['distribution'] = overwrite

    return sweep_config

def sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls):
    from carbs.sweep import WandbCarbs

    carbs_obj = WandbCarbs(args['sweep'])

    while True:
        hparams = carbs_obj.suggest().suggestion
        args['train'].update(hparams)

        if args['neptune']:
            run = init_neptune(args, env_name, id=args['exp_id'], tag=args['tag'])
            stats = train(args, make_env, policy_cls, rnn_cls, neptune=run)[0]
        elif args['wandb']:
            run = init_wandb(args, env_name, id=args['exp_id'], tag=args['tag'])
            stats = train(args, make_env, policy_cls, rnn_cls, wandb=run)[0]


        observation = stats['score']
        carbs_obj.observe(input=hparams, output=observation, cost=1)

def train(args, make_env, policy_cls, rnn_cls, eval_frac=0.1, elos={'model_random.pt': 1000},
        vecenv=None, subprocess=False, queue=None, wandb=None, neptune=None):
    if subprocess:
        from multiprocessing import Process, Queue
        queue = Queue()
        p = Process(target=train, args=(args, make_env, policy_cls, rnn_cls, wandb,
            eval_frac, elos, False, queue))
        p.start()
        p.join()
        stats, uptime, elos = queue.get()

    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    elif args['vec'] == 'native':
        vec = pufferlib.environment.PufferEnv
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray/native).')

    if vecenv is None:
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=args['env'],
            num_envs=args['train']['num_envs'],
            num_workers=args['train']['num_workers'],
            batch_size=args['train']['env_batch_size'],
            zero_copy=args['train']['zero_copy'],
            overwork=args['vec_overwork'],
            backend=vec,
        )

    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)

    '''
    if env_name == 'moba':
        import torch
        os.makedirs('moba_elo', exist_ok=True)
        torch.save(policy, os.path.join('moba_elo', 'model_random.pt'))
    '''

    train_config = pufferlib.namespace(**args['train'], env=env_name,
        exp_id=args['exp_id'] or env_name + '-' + str(uuid.uuid4())[:8])
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb, neptune=neptune)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    uptime = data.profile.uptime
    steps_evaluated = 0
    steps_to_eval = int(args['train']['total_timesteps'] * eval_frac)
    batch_size = args['train']['batch_size']
    while steps_evaluated < steps_to_eval:
        stats, _ = clean_pufferl.evaluate(data)
        steps_evaluated += batch_size

    clean_pufferl.mean_and_log(data)

    '''
    if env_name == 'moba':
        exp_n = len(elos)
        model_name = f'model_{exp_n}.pt'
        torch.save(policy, os.path.join('moba_elo', model_name))
        from evaluate_elos import calc_elo
        elos = calc_elo(model_name, 'moba_elo', elos)
        stats['elo'] = elos[model_name]
        if wandb is not None:
            wandb.log({'environment/elo': elos[model_name]})
    '''

    clean_pufferl.close(data)
    if queue is not None:
        queue.put((stats, uptime, elos))

    return stats, uptime, elos, vecenv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--env', '--environment', type=str,
        default='puffer_squared', help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train',
        choices='train eval evaluate sweep sweep-carbs autotune profile'.split())
    parser.add_argument('--vec', '--vector', '--vectorization', type=str,
        default='native', choices=['serial', 'multiprocessing', 'ray', 'native'])
    parser.add_argument('--vec-overwork', action='store_true',
        help='Allow vectorization to use >1 worker/core. Not recommended.')
    parser.add_argument('--eval-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--baseline', action='store_true',
        help='Load pretrained model from WandB if available')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--exp-id', '--exp-name', type=str,
        default=None, help='Resume from experiment')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.add_argument('--wandb', action='store_true', help='Track on WandB')
    parser.add_argument('--neptune', action='store_true', help='Track on Neptune')
    #parser.add_argument('--wandb-project', type=str, default='pufferlib')
    #parser.add_argument('--wandb-group', type=str, default='debug')
    args = parser.parse_known_args()[0]

    file_paths = glob.glob('config/**/*.ini', recursive=True)
    for path in file_paths:
        p = configparser.ConfigParser()
        p.read('config/default.ini')

        subconfig = os.path.join(*path.split('/')[:-1] + ['default.ini'])
        if subconfig in file_paths:
            p.read(subconfig)

        p.read(path)
        if args.env in p['base']['env_name'].split():
            break
    else:
        raise Exception('No config for env_name {}'.format(args.env))

    for section in p.sections():
        for key in p[section]:
            argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    parsed = parser.parse_args().__dict__
    args = {'env': {}, 'policy': {}, 'rnn': {}}
    env_name = parsed.pop('env')
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    package = args['base']['package']
    module_name = f'pufferlib.environments.{package}'
    if package == 'ocean':
        module_name = 'pufferlib.ocean'

    import importlib
    env_module = importlib.import_module(module_name)

    make_env = env_module.env_creator(env_name)
    policy_cls = getattr(env_module.torch, args['base']['policy_name'])
    
    rnn_name = args['base']['rnn_name']
    rnn_cls = None
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['base']['rnn_name'])

    if args['baseline']:
        assert args['mode'] in ('train', 'eval', 'evaluate')
        args['track'] = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args['exp_id'] = f'puf-{version}-{env_name}'
        args['wandb_group'] = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args["exp_id"]}', ignore_errors=True)
        run = init_wandb(args, args['exp_id'], resume=False)
        if args['mode'] in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args['eval_model_path'] = os.path.join(data_dir, model_file)
    if args['mode'] == 'train':
        wandb = None
        if args['track']:
            run = init_neptune(args, env_name, id=args['exp_id'])
        train(args, make_env, policy_cls, rnn_cls, neptune=run)
    elif args['mode'] in ('eval', 'evaluate'):
        vec = pufferlib.vector.Serial
        if args['vec'] == 'native':
            vec = pufferlib.environment.PufferEnv

        clean_pufferl.rollout(
            make_env,
            args['env'],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            backend=vec,
            model_path=args['eval_model_path'],
            render_mode=args['render_mode'],
            device=args['train']['device'],
        )
    elif args['mode'] == 'sweep':
        assert args['wandb'] or args['neptune'], 'Sweeps require either wandb or neptune'
        sweep(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'sweep-carbs':
        sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
    elif args['mode'] == 'profile':
        import cProfile
        cProfile.run('train(args, make_env, policy_cls, rnn_cls, wandb=None)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)


