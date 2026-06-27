"""Persistent league state and batch Elo fitting for policy sweeps."""

from __future__ import annotations

import contextlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from copy import deepcopy
from typing import Any, Callable

import numpy as np


STATE_VERSION = 1
ANCHOR_ID = "random"
LOGISTIC_TO_ELO = 400.0 / math.log(10.0)
SWEEP_CONTROL_KEYS = (
    'league',
    'league_match_games',
    'league_match_eval_agents',
    'league_anchor_prob',
    'league_state_path',
)


def pair_key(a_id: str, b_id: str) -> str:
    a, b = sorted((a_id, b_id))
    return f"{a}||{b}"


def make_state(sweep_id: str, arch: dict[str, int] | None = None,
        config: dict[str, Any] | None = None) -> dict[str, Any]:
    now = time.time()
    return {
        "version": STATE_VERSION,
        "sweep_id": sweep_id,
        "created_at": now,
        "updated_at": now,
        "anchor_id": ANCHOR_ID,
        "arch": deepcopy(arch or {}),
        "config": deepcopy(config or {}),
        "players": [],
        "matches": [],
        "ratings": {},
    }


@contextlib.contextmanager
def _state_lock(path: str):
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except ImportError:
            pass
        try:
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except ImportError:
                pass


def read_state(path: str, retries: int = 3, delay: float = 0.05) -> dict[str, Any] | None:
    for attempt in range(retries):
        try:
            with open(path) as f:
                state = json.load(f)
            if int(state.get("version", 0)) != STATE_VERSION:
                raise RuntimeError(f"Unsupported league state version in {path}")
            return state
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            if attempt + 1 >= retries:
                raise
            time.sleep(delay)
    return None


def _write_state(path: str, state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state["updated_at"] = time.time()
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def update_state(path: str, mutate: Callable[[dict[str, Any]], Any],
        default_state: dict[str, Any] | None = None) -> Any:
    with _state_lock(path):
        state = read_state(path)
        if state is None:
            if default_state is None:
                raise FileNotFoundError(path)
            state = deepcopy(default_state)
        result = mutate(state)
        _write_state(path, state)
        return result


def load_or_create(path: str, sweep_id: str, arch: dict[str, int] | None = None,
        config: dict[str, Any] | None = None) -> dict[str, Any]:
    default = make_state(sweep_id, arch=arch, config=config)

    def _noop(state):
        return deepcopy(state)

    return update_state(path, _noop, default_state=default)


def _players_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {player["id"]: player for player in state.get("players", [])}


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode('latin1')
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_anchor(path: str, checkpoint_path: str, arch: dict[str, int],
        hypers: dict[str, Any] | None = None) -> dict[str, Any]:
    def _mutate(state):
        players = _players_by_id(state)
        anchor = players.get(ANCHOR_ID)
        if anchor is None:
            anchor = {
                "id": ANCHOR_ID,
                "run_id": ANCHOR_ID,
                "kind": "anchor",
                "checkpoint_path": checkpoint_path,
                "hypers": deepcopy(hypers or {}),
                "arch": deepcopy(arch),
                "cost": 0.0,
                "elo": 0.0,
                "games": 0,
                "matches": 0,
                "score_sum": 0.0,
                "created_at": time.time(),
            }
            state.setdefault("players", []).append(anchor)
        else:
            anchor.update({
                "checkpoint_path": checkpoint_path,
                "hypers": deepcopy(hypers or anchor.get("hypers", {})),
                "arch": deepcopy(arch),
                "elo": 0.0,
            })
        state["arch"] = {"mode": "per_player", "anchor": deepcopy(arch)}
        recompute_ratings_in_state(state)
        return deepcopy(anchor)

    return update_state(path, _mutate)


def register_player(path: str, run_id: str, checkpoint_path: str,
        hypers: dict[str, Any], cost: float, arch: dict[str, int] | None = None) -> dict[str, Any]:
    def _mutate(state):
        players = _players_by_id(state)
        player = players.get(run_id)
        row = {
            "id": run_id,
            "run_id": run_id,
            "kind": "policy",
            "checkpoint_path": checkpoint_path,
            "hypers": _jsonable(deepcopy(hypers)),
            "cost": float(cost),
            "created_at": time.time(),
        }
        if arch is not None:
            row["arch"] = deepcopy(arch)
        if player is None:
            row.update({
                "elo": 0.0,
                "games": 0,
                "matches": 0,
                "score_sum": 0.0,
            })
            state.setdefault("players", []).append(row)
            player = row
        else:
            player.update(row)
        recompute_ratings_in_state(state)
        return deepcopy(player)

    return update_state(path, _mutate)


def opponent_pool(state: dict[str, Any]) -> list[dict[str, Any]]:
    players = []
    for player in state.get("players", []):
        checkpoint_path = player.get("checkpoint_path")
        if not checkpoint_path:
            continue
        if player.get("kind") not in ("anchor", "policy", "random"):
            continue
        players.append({
            "id": player["id"],
            "run_id": player.get("run_id", player["id"]),
            "path": checkpoint_path,
            "elo": float(player.get("elo", 0.0)),
            "kind": player.get("kind", "policy"),
            "hypers": deepcopy(player.get("hypers", {})),
            "arch": deepcopy(player.get("arch", {})),
        })
    return players


def run_id_scores(state: dict[str, Any]) -> dict[str, float]:
    scores = {}
    for player in state.get("players", []):
        if player.get("kind") != "policy":
            continue
        run_id = player.get("run_id") or player.get("id")
        if run_id:
            scores[run_id] = float(player.get("elo", 0.0))
    return scores


def _match_edges(matches: list[dict[str, Any]]) -> list[tuple[str, str, float, float]]:
    edges = []
    for match in matches:
        games = float(match.get("games", 0.0))
        if games <= 0:
            continue
        a_id = match.get("a") or match.get("a_id")
        b_id = match.get("b") or match.get("b_id")
        if not a_id or not b_id or a_id == b_id:
            continue
        score = float(match.get("a_score_rate", match.get("a_score", 0.5)))
        score = min(max(score, 1e-6), 1.0 - 1e-6)
        edges.append((a_id, b_id, score, games))
    return edges


def recompute_ratings(players: list[dict[str, Any]],
        matches: list[dict[str, Any]], anchor_id: str = ANCHOR_ID) -> dict[str, float]:
    ids = [player["id"] for player in players]
    if not ids:
        return {}

    variable_ids = [pid for pid in ids if pid != anchor_id]
    var_index = {pid: idx for idx, pid in enumerate(variable_ids)}
    edges = _match_edges(matches)
    ratings = {pid: 0.0 for pid in ids}
    if not edges or not variable_ids:
        ratings[anchor_id] = 0.0
        return ratings

    def beta(beta_vec: np.ndarray, pid: str) -> float:
        if pid == anchor_id:
            return 0.0
        idx = var_index.get(pid)
        return 0.0 if idx is None else float(beta_vec[idx])

    def objective(beta_vec: np.ndarray) -> float:
        loss = 0.0
        for a_id, b_id, score, games in edges:
            z = np.clip(beta(beta_vec, a_id) - beta(beta_vec, b_id), -30.0, 30.0)
            prob = 1.0 / (1.0 + math.exp(-z))
            loss -= games * (score * math.log(prob) + (1.0 - score) * math.log(1.0 - prob))
        # Keeps disconnected components finite and centered without moving the anchor.
        loss += 1e-3 * float(np.dot(beta_vec, beta_vec))
        return loss

    try:
        from scipy.optimize import minimize
        result = minimize(objective, np.zeros(len(variable_ids)), method="L-BFGS-B")
        beta_vec = result.x if result.success or result.x is not None else np.zeros(len(variable_ids))
    except Exception:
        beta_vec = np.zeros(len(variable_ids), dtype=np.float64)
        lr = 0.01 / max(sum(edge[3] for edge in edges), 1.0)
        for _ in range(1000):
            grad = np.zeros_like(beta_vec)
            for a_id, b_id, score, games in edges:
                z = np.clip(beta(beta_vec, a_id) - beta(beta_vec, b_id), -30.0, 30.0)
                prob = 1.0 / (1.0 + math.exp(-z))
                delta = games * (prob - score)
                if a_id in var_index:
                    grad[var_index[a_id]] += delta
                if b_id in var_index:
                    grad[var_index[b_id]] -= delta
            grad += 2e-3 * beta_vec
            beta_vec -= lr * grad

    for pid in variable_ids:
        ratings[pid] = float(beta_vec[var_index[pid]] * LOGISTIC_TO_ELO)
    ratings[anchor_id] = 0.0
    return ratings


def recompute_ratings_in_state(state: dict[str, Any]) -> None:
    players = state.setdefault("players", [])
    matches = state.setdefault("matches", [])
    ratings = recompute_ratings(players, matches, state.get("anchor_id", ANCHOR_ID))
    stats = {
        player["id"]: {"games": 0, "matches": 0, "score_sum": 0.0}
        for player in players
    }
    for match in matches:
        games = int(match.get("games", 0))
        if games <= 0:
            continue
        a_id = match.get("a") or match.get("a_id")
        b_id = match.get("b") or match.get("b_id")
        if a_id not in stats or b_id not in stats:
            continue
        score = float(match.get("a_score_rate", match.get("a_score", 0.5)))
        stats[a_id]["games"] += games
        stats[b_id]["games"] += games
        stats[a_id]["matches"] += 1
        stats[b_id]["matches"] += 1
        stats[a_id]["score_sum"] += score * games
        stats[b_id]["score_sum"] += (1.0 - score) * games

    for player in players:
        pid = player["id"]
        player["elo"] = float(ratings.get(pid, 0.0))
        player.update(stats.get(pid, {"games": 0, "matches": 0, "score_sum": 0.0}))
    state["ratings"] = ratings


def record_match(path: str, a_id: str, b_id: str, games: int,
        a_score_rate: float, draw_rate: float = 0.0) -> dict[str, float]:
    def _mutate(state):
        state.setdefault("matches", []).append({
            "a": a_id,
            "b": b_id,
            "games": int(games),
            "a_score_rate": float(a_score_rate),
            "draw_rate": float(draw_rate),
            "time": time.time(),
        })
        recompute_ratings_in_state(state)
        return deepcopy(state.get("ratings", {}))

    return update_state(path, _mutate)


def pair_counts(matches: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in matches:
        a_id = match.get("a") or match.get("a_id")
        b_id = match.get("b") or match.get("b_id")
        if not a_id or not b_id:
            continue
        key = pair_key(a_id, b_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def choose_match_pair(state: dict[str, Any], rng: random.Random | None = None,
        anchor_prob: float = 0.12, temperature: float = 150.0,
        neighbor_window: int = 12) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = rng or random
    players = opponent_pool(state)
    policies = [p for p in players if p.get("kind") == "policy"]
    anchor = next((p for p in players if p["id"] == state.get("anchor_id", ANCHOR_ID)), None)
    if not policies:
        raise RuntimeError("Need at least one completed policy for league matches")

    by_id = {p["id"]: p for p in state.get("players", [])}
    counts = pair_counts(state.get("matches", []))

    def matches(pid: str) -> int:
        return int(by_id.get(pid, {}).get("matches", 0))

    if anchor is not None and rng.random() < anchor_prob:
        weights = [1.0 / math.sqrt(matches(p["id"]) + 1.0) for p in policies]
        return anchor, rng.choices(policies, weights=weights, k=1)[0]

    if len(policies) == 1:
        if anchor is None:
            raise RuntimeError("Need at least two policies or an anchor for league matches")
        return policies[0], anchor

    weights = [1.0 / math.sqrt(matches(p["id"]) + 1.0) for p in policies]
    a = rng.choices(policies, weights=weights, k=1)[0]
    candidates = [p for p in policies if p["id"] != a["id"]]
    candidates.sort(key=lambda p: abs(float(p.get("elo", 0.0)) - float(a.get("elo", 0.0))))
    candidates = candidates[:max(1, int(neighbor_window))]
    cand_weights = []
    for b in candidates:
        dist = abs(float(a.get("elo", 0.0)) - float(b.get("elo", 0.0)))
        dist_w = math.exp(-dist / max(float(temperature), 1e-6))
        count_w = 1.0 / math.sqrt(counts.get(pair_key(a["id"], b["id"]), 0) + 1.0)
        cand_weights.append(dist_w * count_w)
    return a, rng.choices(candidates, weights=cand_weights, k=1)[0]


# Protein integration helpers. Kept here so league-specific mutable Elo refresh
# logic is not embedded in the generic sweep optimizer implementation.
def store_protein_score(protein, score):
    if protein.metric_distribution == 'percentile':
        return protein.logit_transform(score)
    return score


def rebuild_protein_top_observations(protein):
    protein.top_observations = sorted(
        protein.success_observations, key=lambda x: x['output'], reverse=True
    )[:protein.num_keep_top_obs]


def refresh_protein_observations_by_run_id(protein, scores_by_run_id):
    updates = 0
    for obs in protein.success_observations:
        run_id = obs.get('run_id')
        if run_id is None or run_id not in scores_by_run_id:
            continue
        obs['output'] = store_protein_score(protein, scores_by_run_id[run_id])
        updates += 1
    if updates:
        rebuild_protein_top_observations(protein)
    return updates


def finish_trial(args, run_id, model_path, all_logs, flat_logs, result_queue):
    if not all_logs:
        all_logs.append(flat_logs)
    metrics = {k: [v] for k, v in all_logs[-1].items()}
    log_dir = os.path.join(args['log_dir'], args['env_name'])
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, run_id + '.json'), 'w') as f:
        json.dump({**args, 'metrics': metrics}, f)
    if args['wandb']:
        import wandb
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

def policy_arch(args):
    return {
        'hidden_size': int(float(args['policy']['hidden_size'])),
        'num_layers': int(float(args['policy']['num_layers'])),
    }


def resolve_state_path(env_name, args):
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


def validate_and_force_config(env_name, args, pareto=False):
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
    sweep_cfg['metric'] = 'elo'
    sweep_cfg['downsample'] = 1


def configure_trial_args(args):
    args.setdefault('selfplay', {})['enabled'] = 1
    # League sweeps use the league only for post-hoc Elo scoring. Each trial
    # remains a reproducible ordinary historical-selfplay run.
    args['vec']['frozen_bank_hidden_size'] = int(float(args['policy']['hidden_size']))
    args['vec']['frozen_bank_num_layers'] = int(float(args['policy']['num_layers']))
    args.setdefault('env', {})['num_agents'] = 2
    args['env']['num_bots'] = 0


def materialize_anchor(env_name, args, state_path, sweep_id, gpu_id, resolve_backend, native_backend):
    arch = policy_arch(args)
    state = read_state(state_path)
    if state is not None:
        for player in state.get('players', []):
            if player.get('id') == ANCHOR_ID and player.get('checkpoint_path'):
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
        backend = resolve_backend(cfg)
        if backend is not native_backend:
            raise RuntimeError('league random anchor creation requires the native CUDA backend')
        pufferl = backend.create_pufferl(cfg)
        backend.save_weights(pufferl, anchor_path)

    ensure_anchor(state_path, anchor_path, arch, hypers={'policy': deepcopy(args['policy'])})
    return anchor_path


def refresh_observations(sweep_obj, state_path):
    state = read_state(state_path)
    if state is None or not hasattr(sweep_obj, 'refresh_observations_by_run_id'):
        return 0
    return sweep_obj.refresh_observations_by_run_id(run_id_scores(state))


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


def apply_match_reward_conditioning(match_args, player_a, player_b):
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


def apply_match_policy_arch(match_args, player_a, player_b):
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


def _match_once_child(env_name, player_a, player_b, games, args, result_queue, match_fn):
    try:
        match_args = deepcopy(args)
        match_args['match_eval_agents'] = int(args['sweep'].get('league_match_eval_agents', 8192))
        match_args['skip_match_close'] = True
        apply_match_policy_arch(match_args, player_a, player_b)
        apply_match_reward_conditioning(match_args, player_a, player_b)
        logs = match_fn(env_name, player_a['path'], player_b['path'],
            num_games=int(games), args=match_args, verbose=False)
        result_queue.put({
            'ok': True,
            'score': float(logs.get('env/slot_0_score', 0.0)),
            'draw': float(logs.get('env/draw_rate', 0.0)),
            'games': int(logs.get('env/n', games)),
        })
    except BaseException as e:
        result_queue.put({'ok': False, 'error': f'{type(e).__name__}: {e}'})


def _match_once(env_name, player_a, player_b, games, args, match_fn):
    ctx = mp.get_context('spawn')
    result_queue = ctx.SimpleQueue()
    proc = ctx.Process(target=_match_once_child,
        args=(env_name, player_a, player_b, int(games), args, result_queue, match_fn))
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


def _run_pair(env_name, player_a, player_b, games, args, match_fn):
    games = int(games)
    if player_a['id'] == player_b['id']:
        return 0.5, 1.0, games
    if games < 2:
        return _match_once(env_name, player_a, player_b, games, args, match_fn)

    games_ab = games // 2
    games_ba = games - games_ab
    score_ab, draw_ab, n_ab = _match_once(env_name, player_a, player_b, games_ab, args, match_fn)
    score_ba, draw_ba, n_ba = _match_once(env_name, player_b, player_a, games_ba, args, match_fn)
    total = max(n_ab + n_ba, 1)
    a_score = (score_ab * n_ab + (1.0 - score_ba) * n_ba) / total
    draw = (draw_ab * n_ab + draw_ba * n_ba) / total
    return a_score, draw, total


def _match_worker(env_name, args, state_path, gpu_id, stop_event, match_fn):
    worker_args = deepcopy(args)
    worker_args['gpu_id'] = gpu_id
    rng = random.Random(int(args.get('seed', 0)) + 1009 * (gpu_id + 1))
    games = int(args['sweep'].get('league_match_games', 4096))
    anchor_prob = float(args['sweep'].get('league_anchor_prob', 0.12))
    while not stop_event.is_set():
        try:
            state = read_state(state_path)
            if state is None:
                if stop_event.wait(2.0):
                    break
                continue
            player_a, player_b = choose_match_pair(
                state, rng=rng, anchor_prob=anchor_prob)
            a_score, draw, total = _run_pair(
                env_name, player_a, player_b, games, worker_args, match_fn)
            ratings = record_match(
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


def sweep(env_name, args=None, pareto=False, *, load_config, validate_config, train, match, resolve_backend, native_backend):
    args = args or load_config(env_name)
    sweep_gpus = args['sweep']['gpus'] or len(os.listdir('/proc/driver/nvidia/gpus'))
    validate_and_force_config(env_name, args, pareto=pareto)
    exp_gpus = int(args['train']['gpus'])

    if sweep_gpus <= 1:
        raise ValueError('league sweep requires at least one training GPU and one match GPU')
    train_slots = sweep_gpus - 1
    train_gpu_ids = list(range(train_slots))
    match_gpu_ids = [sweep_gpus - 1]
    args['no_model_upload'] = True

    state_path, sweep_id = resolve_state_path(env_name, args)
    arch = policy_arch(args)
    load_or_create(state_path, sweep_id, arch=arch, config={
        'env_name': env_name,
        'league_match_games': int(args['sweep'].get('league_match_games', 4096)),
        'trial_opponents': 'historical_selfplay_only',
    })
    materialize_anchor(env_name, args, state_path, sweep_id, match_gpu_ids[0], resolve_backend, native_backend)
    configure_trial_args(args)

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
    match_proc = ctx.Process(target=_match_worker,
        args=(env_name, deepcopy(args), state_path, match_gpu_ids[0], stop_event, match))
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
            player_arch = policy_arch(player_hypers)
            player = register_player(
                state_path, run_id, result['checkpoint_path'], player_hypers, cost, arch=player_arch)
            sweep_obj.observe(done_args, float(player.get('elo', 0.0)), cost,
                is_failure=False, run_id=run_id)
            refresh_observations(sweep_obj, state_path)
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
            refresh_observations(sweep_obj, state_path)
            if idx > 1:
                sweep_obj.suggest(args)
            configure_trial_args(args)
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
            state = read_state(state_path)
            policies = [p for p in state.get('players', []) if p.get('kind') == 'policy'] if state else []
            if not policies or state.get('matches'):
                break
            time.sleep(1.0)

        stop_event.set()
        match_proc.join(timeout=shutdown_timeout)
        if match_proc.is_alive():
            match_proc.terminate()
            match_proc.join(timeout=5)
