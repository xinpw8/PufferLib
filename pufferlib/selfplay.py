"""Selfplay-pool training: a fraction of envs play primary vs a frozen historical
snapshot, the rest are pure selfplay. Used by `_train` in pufferl.py — gated on
`selfplay.enabled` (config section).

Pool growth and opponent swaps are decoupled:
  - snapshot_interval: every N global steps, rank 0 saves primary weights as a
    new pool entry and publishes the pool. 0 disables interval snapshotting.
  - opp_timeout_steps: every N global steps per bank, ranks sample new opponents
    on a fixed cadence and load them after all historical envs reach an episode
    boundary. 0 disables fixed-interval swapping.

Winrate is diagnostic only; it does not trigger snapshots or swaps.
Pool storage is disk-only (paths held in memory; weights only on GPU when
loaded as the frozen bank). Stride-eviction preserves temporal coverage when
the pool exceeds its cap.
"""
import glob
import json
import os
import time

import numpy as np

from pufferlib import _C


def sample_opponent(pool, rng):
    if not pool:
        raise RuntimeError('selfplay opponent pool is empty')
    return pool[int(rng.integers(len(pool)))]


def resolve_opponent_pool(spec):
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        candidates = []
        for item in spec:
            candidates.extend(resolve_opponent_pool(item))
        return sorted(dict.fromkeys(candidates))

    raw_spec = os.path.expanduser(str(spec))

    def _expand(path_spec):
        if os.path.isdir(path_spec):
            return glob.glob(os.path.join(path_spec, '*.bin'))
        candidates = glob.glob(path_spec)
        if not candidates and os.path.isfile(path_spec):
            candidates = [path_spec]
        return candidates

    candidates = _expand(raw_spec)
    if not candidates and not os.path.isabs(raw_spec):
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        candidates = _expand(os.path.join(repo_root, raw_spec))

    return sorted(os.path.abspath(path) for path in candidates
                  if path.endswith('.bin') and os.path.isfile(path))


def evict(pool, max_size):
    '''Drop every other entry from the oldest half once the pool exceeds max_size.
    Newest half is preserved intact.'''
    if len(pool) <= max_size:
        return pool
    half = len(pool) // 2
    return pool[:half:2] + pool[half:]


def build_perm_tags(num_buffers, agents_per_buffer, agents_per_env, frozen_sizes, num_envs):
    '''Build env-slot -> rollout-row routing and per-env bank tag.

    Multi-bank generalization. `frozen_sizes` is a list of per-bank agent counts
    (per buffer). With one bank this reduces to the legacy single-bank layout.

    Per-buffer physical-row layout (apb = agents_per_buffer, F = sum(frozen_sizes)):
        [0,           apb - 2F)                       primary — selfplay envs (all slots)
        [apb - 2F,    apb - F)                        primary — historical envs' team A
        [apb - F,     apb - F + frozen_sizes[0])      bank 0  — historical envs' team B
        [apb - F + frozen_sizes[0], ... + ...[1])     bank 1  — ... etc.

    Env order within a buffer: selfplay envs first (tag=0), then historical
    envs assigned to banks in block order — the first `frozen_sizes[0]/team_size`
    historical envs play bank 0 (tag=1), next block plays bank 1 (tag=2), etc.

    The C-side bank_layout (pufferlib.cu:1798-1806) lays banks out sequentially
    after primary, so our routing matches: bank b's slice is
    [apb - F + sum(frozen_sizes[:b]),  apb - F + sum(frozen_sizes[:b+1])).

    Returns (perm, tags, num_hist_envs_per_bank) — last is a list of per-bank
    historical-env counts across all buffers, used by selfplay.step to know how
    many env alignments to wait for per bank during swaps.'''
    team_size = agents_per_env // 2
    envs_per_buffer = agents_per_buffer // agents_per_env
    num_banks = len(frozen_sizes)
    total_frozen = sum(frozen_sizes)
    hist_envs_per_bank_per_buffer = [fs // team_size for fs in frozen_sizes]
    total_hist_envs_per_buffer = sum(hist_envs_per_bank_per_buffer)
    selfplay_envs = envs_per_buffer - total_hist_envs_per_buffer
    perm = np.empty(num_buffers * agents_per_buffer, dtype=np.int32)
    tags = np.zeros(num_envs, dtype=np.int32)
    env_idx = 0
    for b_buf in range(num_buffers):
        buf_start          = b_buf * agents_per_buffer
        hist_primary_start = buf_start + agents_per_buffer - 2 * total_frozen
        bank_starts = []
        offset = buf_start + agents_per_buffer - total_frozen
        for bank in range(num_banks):
            bank_starts.append(offset)
            offset += frozen_sizes[bank]
        h_within_buffer = 0
        for e in range(envs_per_buffer):
            slot_base = buf_start + e * agents_per_env
            if e < selfplay_envs:
                for s in range(agents_per_env):
                    perm[slot_base + s] = slot_base + s
                tags[env_idx] = 0
            else:
                # Block assignment: walk cumulative bank capacity to find which
                # bank this historical env belongs to.
                bank_idx = 0
                cum = hist_envs_per_bank_per_buffer[0]
                while h_within_buffer >= cum and bank_idx < num_banks - 1:
                    bank_idx += 1
                    cum += hist_envs_per_bank_per_buffer[bank_idx]
                h_in_bank = h_within_buffer - (cum - hist_envs_per_bank_per_buffer[bank_idx])
                team_a_offset = hist_primary_start + h_within_buffer * team_size
                team_b_offset = bank_starts[bank_idx] + h_in_bank * team_size
                for s in range(team_size):
                    perm[slot_base + s] = team_a_offset + s
                    perm[slot_base + team_size + s] = team_b_offset + s
                tags[env_idx] = bank_idx + 1
                h_within_buffer += 1
            env_idx += 1
    num_hist_envs_per_bank = [n * num_buffers for n in hist_envs_per_bank_per_buffer]
    return perm, tags, num_hist_envs_per_bank


def make_bank_state(path, current_agent_step, num_hist_envs):
    return {
        'cur_opp_path': path,
        'hist_score': 0.0,
        'hist_n': 0.0,
        'pending_opp_path': None,
        'epoch_armed': 0,
        'opp_started_step': current_agent_step,
        'num_hist_envs': num_hist_envs,
        'last_winrate_at_swap': 0.0,
        'last_epochs_to_align': 0,
    }


def shared_state_path(pool_dir):
    return os.path.join(pool_dir, 'shared_opponent.json')


def publish_shared_state(pool_state):
    path = pool_state['shared_state_path']
    version = int(pool_state.get('shared_state_version', 0)) + 1
    pool = pool_state.get('pool', ())
    data = {
        'version': version,
        'agent_step': pool_state.get('last_snapshot_step', 0),
        'pool_size': len(pool),
        'pool': [{'path': entry['path']} for entry in pool],
        # Diagnostic + bootstrap for ranks that are just starting. Followers do
        # not keep following rank 0's bank after setup; they sample locally.
        'banks': [{'path': bank['cur_opp_path']} for bank in pool_state['banks']],
    }
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(data, f)
        f.write('\n')
    os.replace(tmp, path)
    pool_state['shared_state_version'] = version


def read_shared_state(path):
    with open(path) as f:
        return json.load(f)


def wait_shared_state(path, timeout=300.0):
    start = time.time()
    while True:
        try:
            return read_shared_state(path)
        except (FileNotFoundError, json.JSONDecodeError):
            if time.time() - start > timeout:
                raise RuntimeError(f'timed out waiting for shared selfplay state: {path}')
            time.sleep(0.05)


def _pool_from_shared_state(state):
    pool = []
    seen = set()
    for entry in state.get('pool', []):
        path = entry.get('path')
        if not path or path in seen:
            continue
        seen.add(path)
        pool.append({'path': path})
    return pool


def _external_pool_entries(sp):
    spec = sp.get('opponent_pool', '')
    paths = resolve_opponent_pool(spec)
    if spec and not paths:
        raise RuntimeError(f'selfplay.opponent_pool resolved no .bin files: {spec}')
    return [{'path': path} for path in paths]


def sync_shared_pool(pool_state):
    try:
        state = read_shared_state(pool_state['shared_state_path'])
    except (FileNotFoundError, json.JSONDecodeError):
        return

    version = int(state.get('version', 0))
    if version <= int(pool_state.get('shared_state_version', 0)):
        return

    pool = _pool_from_shared_state(state)
    if pool:
        pool_state['pool'] = pool
    pool_state['shared_state_version'] = version


def setup(pufferl, backend, args, run_id, artifact_owner=True):
    '''Wire up agent_perm/tags and bootstrap the frozen bank with the current
    weights so historical envs have an opponent from rollout 1. Returns a
    pool_state dict (or None if disabled).'''
    sp = args.get('selfplay', {})
    if not sp.get('enabled', 0):
        return None
    if backend is not _C:
        raise RuntimeError('selfplay_pool requires the native CUDA backend')

    total_agents = int(args['vec']['total_agents'])
    num_buffers = int(args['vec']['num_buffers'])
    if total_agents % num_buffers != 0:
        raise RuntimeError(f'total_agents ({total_agents}) must be divisible by '
                           f'num_buffers ({num_buffers})')
    agents_per_buffer = total_agents // num_buffers

    num_envs = backend.num_envs(pufferl)
    agents_per_env = total_agents // num_envs
    if agents_per_env % 2 != 0:
        raise RuntimeError(f'agents_per_env ({agents_per_env}) must be even (two equal teams)')
    if agents_per_buffer % agents_per_env != 0:
        raise RuntimeError(f'agents_per_buffer ({agents_per_buffer}) must be divisible by '
                           f'agents_per_env ({agents_per_env})')
    team_size = agents_per_env // 2

    num_banks = int(args['vec'].get('num_frozen_banks', 1))
    if num_banks <= 0:
        raise RuntimeError('selfplay.enabled requires num_frozen_banks >= 1')
    if num_banks > 8:
        raise RuntimeError(f'num_frozen_banks {num_banks} exceeds chess.h CHESS_MAX_BANKS=8')

    # frozen_bank_pct is per-bank (matches C-side: pufferlib.cu:2069). Each bank
    # gets floor(apb * pct) agents, total historical = num_banks * frozen_size.
    frozen_size = int(agents_per_buffer * float(args['vec']['frozen_bank_pct']))
    frozen_size -= frozen_size % team_size
    if frozen_size <= 0:
        raise RuntimeError('selfplay.enabled but frozen_bank_pct rounds to 0 slots '
                           f'after team-size ({team_size}) alignment')
    total_frozen = frozen_size * num_banks
    if total_frozen >= agents_per_buffer // 2:
        raise RuntimeError(f'total_frozen {total_frozen} (= num_banks {num_banks} '
                           f'* per_bank {frozen_size}) >= apb/2 {agents_per_buffer//2}')

    frozen_sizes = [frozen_size] * num_banks
    perm, tags, num_hist_envs_per_bank = build_perm_tags(
        num_buffers, agents_per_buffer, agents_per_env, frozen_sizes, num_envs)
    backend.set_agent_perm(pufferl, perm)
    backend.set_env_tags(pufferl, tags)

    pool_dir = os.path.join(args['checkpoint_dir'], args['env_name'], run_id, 'pool')
    state_path = shared_state_path(pool_dir)

    rank = int(args.get('rank', 0))
    rng = np.random.default_rng(int(sp.get('seed', 0)) + rank)
    world_size = max(1, int(args.get('world_size', 1)))
    current_agent_step = int(pufferl.global_step) * world_size

    pool = []
    banks_state = []
    shared_version = 0
    external_pool = _external_pool_entries(sp)
    if artifact_owner:
        os.makedirs(pool_dir, exist_ok=True)
        if external_pool:
            pool = external_pool
        else:
            bootstrap_path = os.path.join(pool_dir, f'{pufferl.global_step:016d}.bin')
            backend.save_weights(pufferl, bootstrap_path)
            pool = [{'path': bootstrap_path}]
        for b in range(num_banks):
            opp_entry = sample_opponent(pool, rng)
            backend.load_frozen_bank(pufferl, b, opp_entry['path'])
            banks_state.append(make_bank_state(
                opp_entry['path'], current_agent_step, num_hist_envs_per_bank[b]))
    else:
        state = wait_shared_state(state_path)
        shared_version = int(state.get('version', 0))
        pool = _pool_from_shared_state(state)
        bank_entries = state.get('banks', [])
        if len(bank_entries) < num_banks:
            raise RuntimeError(f'shared selfplay state has {len(bank_entries)} banks, expected {num_banks}')
        for b in range(num_banks):
            entry = bank_entries[b]
            opp_path = entry['path']
            backend.load_frozen_bank(pufferl, b, opp_path)
            banks_state.append(make_bank_state(
                opp_path, current_agent_step, num_hist_envs_per_bank[b]))

    pool_state = {
        'pool_dir': pool_dir,
        'shared_state_path': state_path,
        'shared_state_version': shared_version,
        'artifact_owner': artifact_owner,
        'pool': pool,
        'rng': rng,
        'max_size': int(sp['max_size']),
        'snapshot_interval': int(sp.get('snapshot_interval', 1_000_000_000)),
        'opp_timeout_steps': int(sp.get('opp_timeout_steps', 500_000_000)),
        'num_banks': num_banks,
        'banks': banks_state,
        'world_size': world_size,
        'last_snapshot_step': current_agent_step,
    }
    if artifact_owner:
        publish_shared_state(pool_state)
    return pool_state


def sync(pufferl, backend, pool_state):
    if pool_state is None:
        return
    if pool_state.get('artifact_owner', True):
        return
    sync_shared_pool(pool_state)


def step(pufferl, backend, pool_state, flat_logs, epoch):
    if pool_state is None:
        return

    n_window = float(flat_logs.get('env/n', 0.0))
    num_banks = pool_state['num_banks']
    current_agent_step = int(pufferl.global_step) * int(pool_state.get('world_size', 1))

    artifact_owner = pool_state.get('artifact_owner', True)
    if not artifact_owner:
        sync_shared_pool(pool_state)

    # 1. Per-bank score accumulation from the most recent rollout window.
    for b in range(num_banks):
        bank = pool_state['banks'][b]
        hist_score_w = float(flat_logs.get(f'env/hist_score_bank_{b}', 0.0)) * n_window
        hist_n_w     = float(flat_logs.get(f'env/hist_n_bank_{b}',     0.0)) * n_window
        if hist_n_w > 0.0:
            bank['hist_score'] += hist_score_w
            bank['hist_n']     += hist_n_w

    # 2. Global snapshot cadence (shared across banks).
    pool_changed = False
    if (artifact_owner
            and pool_state['snapshot_interval'] > 0
            and current_agent_step - pool_state['last_snapshot_step']
                >= pool_state['snapshot_interval']):
        os.makedirs(pool_state['pool_dir'], exist_ok=True)
        snap_path = os.path.join(pool_state['pool_dir'],
            f'{pufferl.global_step:016d}.bin')
        backend.save_weights(pufferl, snap_path)
        pool_state['pool'].append({'path': snap_path})
        pool_state['pool'] = evict(pool_state['pool'], pool_state['max_size'])
        pool_state['last_snapshot_step'] = current_agent_step
        pool_changed = True

    # 3. Per-bank fixed-interval swap logic. Tags 1..num_banks correspond to
    # bank 0..num_banks-1. Winrate is logged but does not drive swaps.
    opponent_changed = False
    for b in range(num_banks):
        bank = pool_state['banks'][b]
        winrate = (bank['hist_score'] / bank['hist_n']
                       if bank['hist_n'] > 0 else None)
        timed_out = (pool_state['opp_timeout_steps'] > 0
            and current_agent_step - bank['opp_started_step']
                >= pool_state['opp_timeout_steps'])
        tag_value = b + 1

        if bank['pending_opp_path'] is not None:
            if backend.count_aligned(pufferl, tag_value, 0) >= bank['num_hist_envs']:
                backend.load_frozen_bank(pufferl, b, bank['pending_opp_path'])
                backend.count_aligned(pufferl, tag_value, 1)
                bank['cur_opp_path'] = bank['pending_opp_path']
                bank['pending_opp_path'] = None
                bank['hist_score'] = 0.0
                bank['hist_n'] = 0.0
                bank['opp_started_step'] = current_agent_step
                bank['last_epochs_to_align'] = epoch - bank['epoch_armed']
                opponent_changed = True
        elif timed_out:
            opp_entry = sample_opponent(pool_state['pool'], pool_state['rng'])
            bank['pending_opp_path'] = opp_entry['path']
            bank['epoch_armed'] = epoch
            bank['last_winrate_at_swap'] = winrate if winrate is not None else 0.0
            # Start a fresh alignment window for the pending opponent.
            backend.count_aligned(pufferl, tag_value, 1)

    if artifact_owner and (pool_changed or opponent_changed):
        publish_shared_state(pool_state)

    # 4. Emit logs — per-bank and aggregate.
    flat_logs['pool/size']     = len(pool_state['pool'])
    flat_logs['pool/num_banks'] = num_banks
    total_score = 0.0
    total_n     = 0.0
    for b in range(num_banks):
        bank = pool_state['banks'][b]
        wr = (bank['hist_score'] / bank['hist_n']
              if bank['hist_n'] > 0 else None)
        flat_logs[f'pool/winrate_at_swap_bank_{b}'] = bank['last_winrate_at_swap']
        flat_logs[f'pool/epochs_to_align_bank_{b}'] = bank['last_epochs_to_align']
        if wr is not None:
            flat_logs[f'pool/winrate_bank_{b}']           = wr
            flat_logs[f'env/historical_winrate_bank_{b}'] = wr
        total_score += bank['hist_score']
        total_n     += bank['hist_n']
    # Aggregate winrate across all banks (legacy compat with old dashboards).
    if total_n > 0:
        agg = total_score / total_n
        flat_logs['pool/winrate']           = agg
        flat_logs['env/historical_winrate'] = agg
