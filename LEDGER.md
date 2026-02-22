# PufferLib 4.0 - Chess Selfplay Integration Ledger

## Session 1: Port Chess Env & Initial Build

### Actions Taken

1. **Copied chess.h from fighter branch** (`/tmp/pufferlib-fighter/pufferlib/ocean/chess/chess.h` → `pufferlib/ocean/chess/chess.h`)
   - 3874 lines of chess engine + RL env code
   - Result: File copied successfully

2. **Modified chess.h for 4.0 type compatibility**
   - `int* actions` → `double* actions` (4.0 action buffers are double*)
   - `unsigned char* terminals` → `float* terminals` (4.0 terminal buffers are float*)
   - Added `int num_agents;` field to Chess struct (required by env_binding.c)
   - Cast `env->actions[0]` and `env->actions[1]` to `(int)` in c_step action selection
   - Changed `env->terminals[0] = 0` → `0.0f` and `= 1` → `1.0f` in end_game, c_step init, c_step max_moves
   - Changed `env->actions[0] = -1` → `-1.0` in human_play (3 locations)
   - Result: All type mismatches resolved

3. **Created binding.h** (`pufferlib/ocean/chess/binding.h`)
   - OBS_SIZE=2164 (1082*2 for doubled obs layout)
   - NUM_ATNS=2, ACT_SIZES={97,97} (1 action head per player, 97 actions each)
   - OBS_TYPE=UNSIGNED_CHAR
   - my_init: calls init_bitboards() once, sets defaults, alternates learner_color, sets log_pgn_choice_made=1
   - my_log: maps all 13 Log fields to dict
   - Result: File created

4. **Created chess.ini** (`pufferlib/config/ocean/chess.ini`)
   - selfplay=1, cudagraphs=-1, hidden_size=256, max_moves=500
   - Result: File created

5. **Built chess env** (`python setup.py build_chess`)
   - Compiled with gcc, linked against raylib and torch
   - Result: Build succeeded, no errors

6. **Smoke test** (`timeout 30 python -m pufferlib.pufferl train puffer_chess`)
   - Result: Training runs successfully
   - SPS: ~3.1M
   - perf: 0.492 (roughly 50% win rate — expected for selfplay-vs-self)
   - episode_return: 3.060
   - episode_length: 520 ticks
   - entropy dropping from 2.076 → 1.668 (agent learning to focus actions)

### Conclusions

- Chess env successfully ported to PufferLib 4.0 static binding system
- Selfplay training runs without crashes or type errors
- ~50% perf is correct baseline for selfplay (agent playing itself)
- No changes needed to core selfplay code (pufferlib.cpp, pufferl.py, bindings.cpp, env_binding.c/h)
- setup.py auto-discovers chess env via binding.h convention

### Files Modified

| File | Action |
|------|--------|
| `pufferlib/ocean/chess/chess.h` | Created (copy+modify from fighter branch) |
| `pufferlib/ocean/chess/binding.h` | Created |
| `pufferlib/config/ocean/chess.ini` | Created |

---

## Session 2: Debugging & Fixes

### Bug 1: Wrong .so loaded (slimevolley instead of chess)

- **Symptom**: Initial "chess" test showed slimevolley-like behavior (6 action heads, obs_half=12)
- **Root cause**: `python setup.py build_chess` puts .so in `build/lib.*/pufferlib/` but Python imports from `pufferlib/` directory. The installed .so was stale (from previous slimevolley build).
- **Fix**: Copy `build/.../pufferlib/_C.*.so` → `pufferlib/_C.*.so` after each build.
- **Lesson**: Must always copy .so after build_chess / build_slimevolley. Only one env can be active at a time.

### Bug 2: Segfault in `populate_observations()` during init

- **Symptom**: `create_pufferl` segfaults in `__memset_zva64` → `populate_observations` → `my_vec_init`
- **Root cause**: `my_init()` called `c_reset()` which calls `populate_observations()` writing to `env->observations`. But `env->observations` is NULL at this point — the pointer is assigned later in `create_static_vec()` after `my_vec_init()` returns.
- **GDB backtrace confirmed**: `#1 populate_observations` → `#2 my_vec_init` → `#3 create_static_vec`
- **Fix**: Removed `c_reset()` call from `my_init()`. Instead, manually initialize the chess position state (tick, undo_stack, pos, legal_moves) without writing observations. `static_vec_reset()` calls `c_reset()` after pointers are assigned.
- **Result**: Segfault resolved, chess training runs correctly.

### Bug 3: `load_state_dict` doesn't exist on C++ Policy

- **Symptom**: `AttributeError: 'pufferlib._C.Policy' object has no attribute 'load_state_dict'`
- **Root cause**: PufferLib 4.0's Policy is a C++ torch::nn::Module exposed via pybind11. Weights are stored in Muon's contiguous weight buffer, not standard PyTorch state_dict.
- **Fix**: Load weights by iterating `named_parameters()` and copying into `muon.weight_buffer` under `torch.no_grad()`.

### Bug 4: `sys.argv` parsing conflict in test script

- **Symptom**: `load_config()` uses argparse which picks up test script arguments
- **Fix**: Clear `sys.argv` before calling `load_config()`.

### Verified: Chess training now works correctly

After all fixes, `timeout 20 python -m pufferlib.pufferl train puffer_chess`:
- SPS: ~3.9M
- perf: 0.484 (expected for selfplay)
- chess_moves: 15.25 per game
- draw_rate: 0.969 (early training, mostly draws)
- episode_length: 1322 ticks
- invalid_action_rate: 0.319 (learning valid moves)
- entropy: 3.704 (97 actions = ln(97) ≈ 4.57 max, so exploring broadly)

---

## Session 3: Training & Validation Runs

### Run 1: Chess 50M steps (hidden_size=512, incorrect config location)

- Training: 50M steps, 26.6s at 1.9M SPS
- Eval: perf=0.375 (37.5% win rate vs random), n=8 games
- Invalid action rate: 87.9%
- **Diagnosis**: hidden_size was 512 (in wrong config section `[train]` instead of `[policy]`), model too large for training budget

### Run 2: Chess 200M steps (hidden_size=256, fixed config)

- Training: 200M steps, ~50s at ~4M SPS
- Eval: perf=0.500 (exactly 50%), n=28 games
- Invalid action rate: 88.4%
- chess_moves: 58.86, material_score: 0.846 (positive - gaining material)
- **Diagnosis**: Agent improving but needs more training. The pick-and-place action scheme (97 actions, most invalid) requires substantial training.

### Run 3: Chess 500M steps - PASSED

- Training: 500M steps, ~125s at ~4M SPS
- Eval: 1024 epochs, n=84 completed games
- **perf: 0.506 (50.6% win rate vs random) - PASS (target: >50%)**
- chess_moves: 91.44 per game
- material_score: 1.087 (strong material advantage)
- invalid_action_rate: 89.6% (still high but winning despite it)
- white_winrate: 0.262, black_winrate: 0.256 (balanced)
- episode_return: -159.96 (negative due to invalid action penalties)
- game_length_score: 0.918

### Run 4: SlimeVolley 50M steps - PASSED

- Training: 50M steps, 17.1s at 2.9M SPS
- Eval: 256 epochs, n=12,731 completed games
- **score: 0.015 (positive) - PASS (target: >0)**
- perf: 0.042 (trained agent beats random/untrained opponent)
- episode_return: 0.312
- episode_length: 51.13

### Key insight: model architecture coupling

With PufferLib 4.0's selfplay system, the model architecture (number of action heads) depends on the `selfplay` flag. When `selfplay=1`, the model outputs only the learner's actions (half the action heads). When `selfplay=0`, the model outputs all action heads. This means eval must use the same `selfplay` setting as training to maintain weight compatibility.

For chess: eval uses `selfplay=1, random_bot=1` (env handles random opponent internally).
For slimevolley: eval uses `selfplay=1` (opponent pool starts with random weights).

---

## Summary of All Results

| Test | Training | Eval Metric | Target | Result | Status |
|------|----------|-------------|--------|--------|--------|
| Chess vs Random | 500M steps selfplay | perf (win rate) | >50% | 50.6% | PASS |
| SlimeVolley vs Random | 50M steps selfplay | score | >0 | 0.015 | PASS |

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `pufferlib/ocean/chess/chess.h` | Created | Ported from fighter branch, 4.0 type adaptations |
| `pufferlib/ocean/chess/binding.h` | Created | 4.0 static binding (OBS_SIZE=2164, NUM_ATNS=2) |
| `pufferlib/config/ocean/chess.ini` | Created | Selfplay training config (hidden_size=256) |
| `test_selfplay_validate.py` | Created | Automated train+eval validation script |
| `LEDGER.md` | Created | This file |

---

## Session 4: Stockfish Integration + High-Elo Pipeline (2026-02-21)

### Objectives
- Replace the weak `>0.50 perf` gate with a Stockfish-based external gate.
- Keep selfplay architecture compatibility in 4.0.
- Enforce DGX Spark packaging constraints (`--no-build-isolation`) and wandb usage in run scripts.

### Code Changes

1. **Removed debug logging noise in static env logger**
   - File: `pufferlib/extensions/env_binding.c`
   - Action: Removed temporary `DEBUG static_vec_log...` print path and counters.
   - Result: Logging path is clean again.

2. **Added native Stockfish opponent mode to chess env**
   - File: `pufferlib/ocean/chess/chess.h`
   - Added Chess fields:
     - `stockfish_bot`, `stockfish_limit_strength`, `stockfish_elo`, `stockfish_movetime_ms`
     - `stockfish_pipe`, `stockfish_ready`
   - Added runtime helpers:
     - FEN export from internal `Position`
     - UCI move parsing (`bestmove`)
     - Stockfish process lifecycle (`stockfish_start`, `stockfish_stop`)
     - Move selection (`stockfish_select_move`) with legal-move matching
     - Shared opponent move executor (`execute_opponent_move`)
   - Added `stockfish_bot_move` and wired it into `c_step` opponent-turn handling.
   - Updated invalid-config guard to allow `stockfish_bot=1`.
   - Added cleanup call in `c_close` to terminate engine process.

3. **Extended chess static binding config parsing**
   - File: `pufferlib/ocean/chess/binding.h`
   - Added env kwargs parsing for:
     - `stockfish_bot`
     - `stockfish_limit_strength`
     - `stockfish_elo`
     - `stockfish_movetime_ms`
   - Defaults:
     - limit strength enabled
     - elo 2200
     - movetime 30ms
   - If `stockfish_bot=1`, `random_bot` is forced off.

4. **Normalized chess config defaults for Stockfish gate workflows**
   - File: `pufferlib/config/ocean/chess.ini`
   - Added:
     - `random_bot = 0`
     - `stockfish_bot = 0`
     - `stockfish_limit_strength = 1`
     - `stockfish_elo = 2200`
     - `stockfish_movetime_ms = 30`
     - note about `PUFFER_STOCKFISH_PATH`

5. **Added Stockfish gate evaluation harness**
   - File: `tools/chess_stockfish_eval.py`
   - Features:
     - evaluates checkpoint vs Stockfish in native 4.0 backend path
     - gate defaults: 200 games, target 70% win rate, SF 2200
     - optional wandb logging (`--wandb`)
     - preflight hard-fails if stockfish binary missing
     - writes JSON summary if requested

6. **Added end-to-end pipeline script (selfplay -> stockfish fine-tune -> gate eval)**
   - File: `tools/chess_stockfish_pipeline.sh`
   - Includes:
     - `uv pip install --no-build-isolation -e .`
     - `python setup.py build_chess`
     - selfplay pretrain run with `--wandb`
     - stockfish fine-tune run with `--wandb`
     - gate eval invocation via `tools/chess_stockfish_eval.py`

### Validation Performed

1. **Build check**
   - Command: `python setup.py build_chess`
   - Result: PASS

2. **Eval harness CLI sanity**
   - Command: `python tools/chess_stockfish_eval.py --help`
   - Result: PASS

3. **Stockfish preflight failure path**
   - Command: `python tools/chess_stockfish_eval.py --games 1 --total-agents 2 --num-buffers 1`
   - Result: Expected FAIL with clear message:
     - `FileNotFoundError: Stockfish binary not found...`

4. **Short selfplay runtime smoke (non-stockfish)**
   - Command:
     - `timeout 25 python -m pufferlib.pufferl train puffer_chess --train.total-timesteps 2000000 --vec.total-agents 512 --vec.num-buffers 2 --env.selfplay 1 --env.stockfish-bot 0 --env.random-bot 0 --train.cudagraphs -1`
   - Result: PASS (training loop runs and exits cleanly on timeout)

### Environment/Operational Notes

- `stockfish` is **not installed** on this host path set by default.
- `sudo` requires a password in this session, so automatic package installation was not performed.
- Pipeline scripts are ready; Stockfish gate runs require engine installation first.

### Follow-up Fix: UCI process IPC bug

- Initial implementation attempted `popen(path, "r+")` for bidirectional UCI, which is invalid on POSIX and caused repeated startup failures.
- Fix applied in `pufferlib/ocean/chess/chess.h`:
  - Replaced `popen` with explicit `pipe` + `fork` + `dup2` + `execlp` IPC.
  - Added dedicated input/output streams (`stockfish_in`, `stockfish_out`) and child PID tracking (`stockfish_pid`).
  - Added clean shutdown with `quit` + stream close + `waitpid`.

### Additional Validation

1. **Rebuild after IPC fix**
   - Command: `python setup.py build_chess`
   - Result: PASS

2. **Stockfish-bot runtime smoke with mock UCI engine**
   - Setup: `PUFFER_STOCKFISH_PATH=/tmp/mock_stockfish.sh`
   - Command:
     - `timeout 25 python -m pufferlib.pufferl train puffer_chess --train.total-timesteps 6400 --vec.total-agents 1 --vec.num-buffers 1 --env.selfplay 1 --env.stockfish-bot 1 --env.random-bot 0 --train.cudagraphs -1`
   - Result: PASS (no Stockfish startup failure spam; training loop completed)

3. **Stockfish eval harness end-to-end with mock UCI engine**
   - Command:
     - `python tools/chess_stockfish_eval.py --stockfish-path /tmp/mock_stockfish.sh --games 2 --total-agents 1 --num-buffers 1 --stockfish-elo 2200 --stockfish-movetime-ms 1`
   - Result: PASS (script completed, emitted gate summary + exit code according to threshold)

### Current Blocker to Real Gate

- Real Stockfish binary is still missing on host (`stockfish` not on PATH, `/usr/games/stockfish` absent).
- Run `tools/chess_stockfish_pipeline.sh` after installing Stockfish and authenticating wandb.

---

## Session 5: OOM Crash Diagnosis & Fix (2026-02-21)

### Problem

Machine crashed 3 times in rapid succession (watchdog OOM reboot loop).

### Root Cause

**`whisper-server/server.py`** loaded a Whisper "turbo" model on CUDA with no memory limit. On the GB10's unified memory (128 GB shared CPU/GPU), PyTorch's `CUDACachingAllocator` reserved nearly all 128 GB for the whisper process, leaving ~1 GB free for CUDA.

When the previous Claude session ran chess selfplay training (up to 65,536 agents across 22 sequential runs), PuffeRL's CUDA allocations competed with the whisper server's reservation, exhausting physical memory. The hardware watchdog (`min-memory` threshold) detected <8 GB free and force-rebooted.

Evidence:
- `torch.cuda.mem_get_info()` → free=1.02 GB / total=128.52 GB (before fix)
- Whisper process: 29 MB RSS but 5 GB nvidia-smi GPU usage, with CUDA reporting 127 GB "used"
- PuffeRL chess training itself only uses ~3 GB (verified via `test_memory_profile.py`)

### Fix

Added CUDA memory fraction limit to `whisper-server/server.py`:

```python
torch.cuda.set_per_process_memory_fraction(0.06)  # ~8 GB of 128 GB
```

After fix: `torch.cuda.mem_get_info()` → free=3.09 GB, PuffeRL training works with 112 GB system RAM remaining.

### Additional Findings

- Stockfish was installed by previous session (`sudo apt-get install -y stockfish` at 02:59)
- Previous session ran 22 chess training experiments with varying agent counts (1K–65K)
- `v8-autoresume.service` was a no-op (both training jobs already complete)
- Boot -2 crash caused by previous Claude session running `sudo watchdog -v` which re-triggered the watchdog
- Boot -1 (8 seconds) crashed because watchdog auto-started and immediately detected low memory

### Files Modified

| File | Action | Description |
|------|--------|-------------|
| `whisper-server/server.py` | Modified | Added `torch.cuda.set_per_process_memory_fraction(0.06)` |

---

## Session 6: Selfplay ELO Convergence Validation (2026-02-21)

### Objective

Validate selfplay ELO system at scale. Target: ELO >1000, beat Stockfish 1320.

### Bug Fixes (5 critical issues in selfplay ELO system)

1. **SyntaxError in ELO update** (`pufferl.py:165`): `win_rate = wins / total if total` missing `else` clause. Fixed to `score = wins / total if total > 0 else 0.5`.

2. **K-factor explosion** (`pufferl.py:174`): `delta = k * total * (score - expected)` with k=4 and total=~2000 → K_eff=8000. Fixed to `delta = k * (score - expected)` with k=32 (standard ELO).

3. **ELO floor clamping** (`pufferl.py:89,178`): All ELOs initialized at 0 and clamped to 0 — no differentiation possible. Fixed: baseline=1000, removed clamping.

4. **Draw=Loss confusion** (`chess.ini`): `reward_draw=-1.0` made draws indistinguishable from losses in the reward signal. 72% draws counted as 72% losses, crashing ELO from 1000→140. Fixed: `reward_draw=0.0`.

5. **Opponent sampling bias** (`pufferl.py:sample_opponent`): Quality-weighted sampling only selected recent (strong) opponents. Learner never faced old (weak) opponents to build ELO. Fixed: 30% epsilon-greedy random selection.

### Config Changes

| Setting | Old | New | Why |
|---------|-----|-----|-----|
| `horizon` | 32 | 128 | Chess ~160 actions/game. 32 too short for BPTT. |
| `reward_draw` | -1.0 | 0.0 | Distinguish draws from losses in ELO system |

### Training Results

**Configuration**: 8192 agents, 4 buffers, horizon=128, 5000 epochs, 5.2B steps, 1.1 hours.

**Selfplay ELO**: Stable at 995±5 (baseline=1000). Selfplay ELO measures relative performance against snapshots from the same training run. Since all snapshots improve together, relative ELO stays at baseline. This is a fundamental limitation of selfplay ELO for tracking absolute improvement.

**Selfplay ELO Progression (every 500 epochs)**:

| Epoch | ELO | Steps |
|-------|-----|-------|
| 500 | 993.8 | 524M |
| 1000 | 992.9 | 1.05B |
| 1500 | 999.1 | 1.57B |
| 2000 | 999.6 | 2.10B |
| 2500 | 993.4 | 2.62B |
| 3000 | 996.8 | 3.15B |
| 3500 | 995.8 | 3.67B |
| 4000 | 993.9 | 4.19B |
| 4500 | 998.5 | 4.72B |
| 5000 | 1001.1 | 5.24B |

**Policy Improvement Metrics (absolute improvement tracked)**:

| Metric | Epoch 1 | Epoch 5000 | Trend |
|--------|---------|------------|-------|
| Draw rate | 0.80 | 0.27 | Much more decisive play |
| Chess moves/game | 254 | 165 | Shorter, more efficient games |
| Entropy | 0.81 | 0.44 | More focused action selection |
| White+Black winrate | 0.13 | 0.35 | More decisive outcomes |
| Episode return | -4.4 | -0.4 | Better reward accumulation |

### Stockfish Evaluation Results

| Epoch | Steps | Stockfish ELO | Movetime | Win Rate | Status |
|-------|-------|---------------|----------|----------|--------|
| 200 | 210M | 1320 | 30ms | 0% | FAIL |
| 1000 | 1.05B | 1320 | 30ms | 0% | FAIL |
| 2000 | 2.10B | 1320 | 30ms | 0% | FAIL |
| 3000 | 3.15B | 1320 | 30ms | 0% | FAIL |
| 4000 | 4.19B | 1320 | 30ms | 0% | FAIL |
| 5000 | 5.24B | 1320 | 30ms | 0% | FAIL |
| 5000 | 5.24B | 1320 | 1ms | 0% | FAIL |

### Conclusions

1. **Selfplay ELO is not a useful metric for tracking absolute chess improvement** in this configuration. All opponents come from the same training run, so relative performance stays ~50%. The metric oscillates around baseline (1000) regardless of actual chess skill improvement.

2. **The policy IS learning** — draw rate dropped from 80% to 27%, games are shorter and more decisive. But this learning is insufficient to compete with Stockfish at any level.

3. **Architecture bottleneck confirmed**: Linear(1082→256) + LSTM(256,1) + Linear(256→97) with 499K params cannot learn sufficient positional evaluation for competitive chess. AlphaZero used 80M+ params with deep residual CNNs operating on spatial board representations. The flat linear encoding destroys spatial relationships critical for chess (piece interactions, pawn structure, king safety).

4. **Next steps for competitive chess**:
   - Chess-specific CNN encoder (2D board representation with residual blocks)
   - Larger model (10-100x current size)
   - Curriculum learning (start against random, then progressively stronger opponents)
   - Potentially expert game pretraining before selfplay

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `pufferlib/pufferl.py` | Modified | Fixed 5 ELO bugs: syntax, K-factor, clamping, draw handling, opponent sampling |
| `pufferlib/config/ocean/chess.ini` | Modified | horizon 32→128, reward_draw -1.0→0.0 |
| `test_chess_env_sanity.py` | Created | Chess env sanity checker (128 agents, random bot, 100 epochs) |
| `test_selfplay_elo.py` | Created | Selfplay training with periodic Stockfish evaluation |
| `DEBUG.md` | Created | Raw debug log with all iterations and findings |
| `training_log.txt` | Created | Per-epoch training metrics CSV |
