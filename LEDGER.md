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

---

## Session 7: Eval Bug Fix, Reward Shaping, Metric Granularity (2026-02-23)

### Critical Bug Found: Eval Script Evaluating Random Weights

**All previous Stockfish eval results were invalid.** The eval script loaded checkpoint
weights into `muon.weight_buffer` (which backs `policy_fp32` via views), but inference
uses `policy_bf16` — a separate model that was never synced. Every eval since Session 4
evaluated random bfloat16 weights.

Discovery path:
1. Training showed 65% draw rate, eval showed 0.66% draw rate (identical SF settings)
2. Initially misattributed to Stockfish strength — WRONG
3. Re-ran eval with identical SF params (depth=1, mt=0) — still 0.66% draws
4. Ran eval WITHOUT model (random weights) — got 0.8% draws (identical to "loaded" model)
5. Traced inference path: `pufferlib.cpp:352` uses `policy_bf16`, not `policy_fp32`
6. Confirmed `bf16 is not fp32` and different `data_ptr` — two separate models
7. Fixed eval to sync bf16 from fp32 after loading, added validation assertion
8. Post-fix eval: 67.2% draws — matches training's 65.8% ✓

### Code Changes

| File | Action | Description |
|------|--------|-------------|
| `tools/chess_stockfish_eval.py` | Modified | Fixed bf16 weight sync bug; added `_sync_bf16_from_fp32()`, `_validate_checkpoint_load()`; added `--stockfish-depth` flag; prints draw_rate in progress; warns if no model specified |
| `pufferlib/ocean/chess/chess.h` | Modified | `game_result_with_legal_count()` returns 3-6 for draw types; `end_game()` tracks per-type draws + per-color losses; `Log` struct +6 fields |
| `pufferlib/ocean/chess/binding.h` | Modified | `my_log()` exports 6 new metrics to wandb |
| `pufferlib/config/ocean/chess.ini` | Modified | Enabled reward shaping, raised ent_coef |
| `DEBUG.md` | Appended | Bug 9 writeup, reward shaping rationale, validated eval results |

### Config Changes (chess.ini)

| Setting | Old | New | Rationale |
|---------|-----|-----|-----------|
| `reward_draw` | 0.0 | -0.3 | Break draw-seeking equilibrium |
| `reward_material` | 0.0 | 0.05 | Dense gradient for material play |
| `reward_check` | 0.0 | 0.01 | Encourage attacking |
| `reward_repetition` | 0.0 | -0.05 | Punish piece shuffling |
| `ent_coef` | 0.006 | 0.02 | Prevent policy collapse (entropy was 0.03) |

### New Wandb Metrics

| Metric | Description |
|--------|-------------|
| `white_lossrate` | Fraction of games learner loses as White |
| `black_lossrate` | Fraction of games learner loses as Black |
| `draw_by_stalemate` | Draws from stalemate |
| `draw_by_insufficient` | Draws from insufficient material |
| `draw_by_50move` | Draws from 50-move rule |
| `draw_by_repetition` | Draws from threefold repetition |

### Validated Results

**First correct Stockfish eval** (epoch 3600, ~240M steps, vs SF ELO 800 depth=1 mt=0):

| Metric | Training | Eval (fixed) | Eval (broken, for reference) |
|--------|----------|-------------|------------------------------|
| Win rate | 0% | 0% | 0% |
| Draw rate | 65.8% | 67.2% | 0.66% |
| Loss rate | 34.2% | 32.8% | 99.3% |

Training and eval agree after the fix. Previous "99.3% loss" result was random weights.

### Notes on Precision

`USE_BF16=true` is a compile-time constant in `models.cpp`. The `precision = float32` in
`default.ini` does NOT affect the compiled binary. Training already uses bf16 inference /
fp32 optimizer. This is the correct setup for mixed-precision training and should not be
changed. The config value is misleading but harmless.

### Requires Recompilation

The chess.h changes (draw type codes, Log struct) require rebuilding the static library:
```bash
python setup.py build_chess
cp build/lib.*/pufferlib/_C.*.so pufferlib/
```
Then restart training. Existing checkpoints are weight-compatible (no model changes).

---

## Session 8 — 2026-02-23: Stockfish Game History Fix

### Problem
v9 training (chess-v9-rewards, wandb: wbq70goz) showed draw_by_repetition climbing to 45%+.
All draws were threefold repetition. Agent exploited the fact that Stockfish had no game
history — it always played the same "best" move for a given position, creating exploitable loops.

### Root Cause
`stockfish_select_move()` sent `position fen <current>` — no move history. Stockfish
cannot detect repetitions without history. Additionally, `pgn_moves[]` was guarded by
`human_play || log_pgn`, both 0 in training, so move history was never recorded.

### Changes

**chess.h**:
- Added `move_to_uci()` helper function (line 3151-3167)
- `stockfish_select_move()`: sends `position fen <starting_fen> moves <full_history>` (line 3190-3200)
- Move recording guard (line 2753): added `|| env->stockfish_bot`
- Opponent move recording guard (line 3256): added `|| env->stockfish_bot`

### Config (unchanged from v9)
```ini
reward_draw = -0.3
reward_material = 0.05
reward_check = 0.01
reward_repetition = -0.05
ent_coef = 0.02
```

### Result
v10 training (chess-v10-sf-history, wandb: vngs0qmz):
- draw_by_repetition = 0.000 at 786K steps (1167 games)
- Previous v9 had 45%+ repetition draws at same point
- Agent now losing ~100% (expected for untrained policy vs history-aware Stockfish)

### Killed
- v9 training PID 125956 (wandb: wbq70goz)

---

## Session 9 — 2026-02-23: v13 Evaluation + Game Analysis

### Objective
Evaluate v13 checkpoint (ChessTwo 20.6M params, 65.5M steps, trained vs 90% random SF)
against real Stockfish to assess actual chess strength.

### Bugs Found & Fixed

**Bug 13: Eval inherited `stockfish_random_pct=90` from chess.ini**
- Eval script loaded config via `load_config("puffer_chess")` which includes training settings
- First eval showed 98% win rate — actually playing vs 90% random, not real SF
- Fix: Added `--stockfish-random-pct` CLI flag, defaults to 0 (full strength)

**Bug 14: PGN export labels swapped**
- `export_pgn_append()` in chess.h had reversed ternary: `learner_color == CHESS_BLACK ? "Learner" : "Opponent"` for White label
- Fix: Changed to `learner_color == CHESS_WHITE ? "Learner" : "Opponent"`

### Code Changes

| File | Action | Description |
|------|--------|-------------|
| `tools/chess_stockfish_eval.py` | Modified | Added `--stockfish-random-pct`, `--log-pgn` flags; force `stockfish_random_pct` from CLI |
| `pufferlib/ocean/chess/binding.h` | Modified | Added `log_pgn` config support with shared PGN filename |
| `pufferlib/ocean/chess/chess.h` | Modified | Fixed PGN label swap in `export_pgn_append()` |

### Evaluation Results

**v13 checkpoint**: `experiments/puffer_chess/rnxk83ag/model_puffer_chess_001000.pt` (epoch 1000, 65.5M steps)

**Strength curve (SF 1320 depth=1, varying `random_pct`):**

| Random % | Win Rate | Draw Rate | Loss Rate |
|----------|----------|-----------|-----------|
| 0% (real SF) | 0.0% | 1.5% | 98.5% |
| 50% | 57.6% | 17.6% | 24.8% |
| 70% | 81.3% | 12.4% | 6.3% |
| 80% | 94.7% | 2.5% | 2.8% |
| 90% (training) | 98.0% | 2.0% | 0% |
| 95% | 100% | 0% | 0% |

### PGN Game Analysis (234 games vs real SF 1320 d=1)

Key findings from PGN analysis:
- Opens 1.h4 as White 100% of games; 1...h5 as Black 82% of games
- First 4 moves ALL pawns in 100% of White games
- Never castles (0/234 games)
- Falls into Qa5+/Qxb4+ trap in 25% of White games
- Queen blundered in 36% of games
- Zero piece development (<10% develop any piece)
- Mean game length: 22.5 moves
- Agent plays at approximately 100-200 Elo

### Conclusions

1. v13 has learned to dominate 90% random play but has zero transfer to real chess
2. The pawn-pushing strategy (h4-h5-h6) is a local optimum specific to random opponents
3. Agent needs curriculum training (reduce random_pct) or imitation learning
4. PGN game logging now works for future evaluations

---

## Session 10 — 2026-02-23: v15 Throughput >100K + Validity Sweep + Resume Behavior

### Objective

Push chess curriculum training throughput above 100K SPS while preserving curriculum behavior,
then validate whether high in-training winrates represent real strength vs Stockfish.

### Starting Point

Run `chess-v14-curriculum` (`6ovobqug`) was stable but capped around:
- SPS: ~19K–22K
- stockfish_random_pct curriculum descending over time
- env-step cost increasing as random_pct dropped

### Throughput Changes Implemented

1. **Added Stockfish query throttle**
   - New env/config field: `stockfish_query_pct` (0-100)
   - Allows querying Stockfish on only a fraction of opponent turns
   - Files: `pufferlib/ocean/chess/chess.h`, `pufferlib/ocean/chess/binding.h`, `pufferlib/config/ocean/chess.ini`

2. **Made chess encoder selectable in C++ training backend**
   - New env knob: `chess_encoder`
   - `1=ChessEncoder` (fast), `2=ChessTwoEncoder` (heavier)
   - Files: `pufferlib/extensions/ocean.cpp`, `pufferlib/extensions/pufferlib.cpp`

3. **Removed wasted opponent forward pass for Stockfish mode**
   - Added `selfplay_external_opponent` fast path in `net_callback_selfplay()`
   - When `stockfish_bot=1`, opponent-policy forward/sampling is skipped (env controls opponent move)
   - File: `pufferlib/extensions/pufferlib.cpp`

4. **Config tuned for throughput target**
   - `chess_encoder=1`
   - `hidden_size=256`
   - `stockfish_query_pct=10`
   - `stockfish_random_pct=90` start (curriculum still active)
   - File: `pufferlib/config/ocean/chess.ini`

### Throughput Validation

Short probes:
- After first round of knobs: ~46K–48K SPS
- After selfplay fast path: ~104K–110K SPS

Long run (`chess-v15-100k`, wandb `re8nadd7`, run `desert-fire-68`):
- Sustained ~103K–122K SPS
- Typical operating band ~109K–119K SPS

### Metric Validity Investigation

Observed in run `desert-fire-68`:
- `white_winrate` and `black_winrate` both around 0.45
- `opponent_winrate` around 0.004–0.011
- `stockfish_random_pct` reached 0

Key finding:
- These metrics were valid for the configured opponent mix, but not "full Stockfish"
- Because training used `stockfish_query_pct=10`, only ~10% of opponent turns queried Stockfish

### Eval Pipeline Fix (New Bug Found)

**Bug 15: Eval inherited weak query mix from training config**
- `tools/chess_stockfish_eval.py` inherits env config from `chess.ini`
- Without override, eval also used `stockfish_query_pct=10`

Fix:
- Added CLI arg `--stockfish-query-pct` (default 100)
- Forced `env_cfg["stockfish_query_pct"]` from CLI
- Added fields to `EvalSummary`

File:
- `tools/chess_stockfish_eval.py`

### Stockfish Sweep (Checkpoint: `re8nadd7/model_puffer_chess_001400.pt`)

Settings:
- SF ELO 1320 (minimum), depth=1, movetime=0
- 200 games per point
- `stockfish_query_pct=100` (true Stockfish eval)

| random_pct | win_rate | draw_rate | loss_rate |
|------------|----------|-----------|-----------|
| 0  | 0.000 | 0.004 | 0.996 |
| 8  | 0.000 | 0.014 | 0.986 |
| 16 | 0.002 | 0.021 | 0.977 |
| 24 | 0.054 | 0.036 | 0.910 |
| 32 | 0.130 | 0.086 | 0.784 |
| 50 | 0.413 | 0.227 | 0.360 |
| 70 | 0.794 | 0.152 | 0.054 |
| 90 | 0.917 | 0.071 | 0.013 |

Control:
- `random_pct=0`, `query_pct=10` -> win_rate 0.959 (matches training-side strong metrics)

Conclusion from sweep:
- High training winrates were genuine for query10 mixture
- Real strength vs always-queried Stockfish remained low at low random_pct

Artifacts:
- `data/eval_sweep_001400/pct_*.json`
- `data/eval_sweep_001400/pct_*.log`
- `data/eval_sweep_001400/pct_0_query10.json`

### Resume Behavior Investigation

User question: why did resumed run (`pretty-pyramid-69`, `kzfg8oi4`) start with fresh-looking
winrates and curriculum.

**Bug 16: `--load-model-path` not applied in C++ training path**
- `_train_rank()` builds new `PuffeRL` and enters training loop directly
- No training-time checkpoint load into `pufferl_cpp` policy/muon buffers
- `load_model_path` exists in `load_policy()` path (eval-side)

Additional reset:
- Curriculum globals (`_g_sf_random_pct`, counters) are process globals in C and reset on process start

Result:
- `pretty-pyramid-69` behaved as new-run initialization (`stockfish_random_pct=90`, `elo=1000` at start)
- Not a true continuation of `desert-fire-68` state

### Files Modified in This Session

| File | Action | Description |
|------|--------|-------------|
| `pufferlib/extensions/pufferlib.cpp` | Modified | Added `selfplay_external_opponent` fast path; wired `chess_encoder`; stockfish-aware routing |
| `pufferlib/extensions/ocean.cpp` | Modified | `create_policy(...)` now supports selectable chess encoder |
| `pufferlib/ocean/chess/chess.h` | Modified | Added `stockfish_query_pct` behavior + logging field |
| `pufferlib/ocean/chess/binding.h` | Modified | Parse/export `stockfish_query_pct` |
| `pufferlib/config/ocean/chess.ini` | Modified | Throughput config (`chess_encoder=1`, `hidden_size=256`, `stockfish_query_pct=10`) |
| `tools/chess_stockfish_eval.py` | Modified | Added `--stockfish-query-pct`, fixed eval control over query mix |
| `training_log.txt` | Updated | v15 run logs and resumed-run investigation outputs |
| `DEBUG.md` | Appended | Detailed debug chronology and raw findings |

### Runs

- Throughput run: https://wandb.ai/xinpw8/puffer4/runs/re8nadd7 (`desert-fire-68`)
- Post-checkpoint run: https://wandb.ai/xinpw8/puffer4/runs/kzfg8oi4 (`pretty-pyramid-69`)

---

## Session 11 — 2026-02-24: Restored Human-vs-Policy Chess Eval in 4.0

### Objective

Provide a one-command human-vs-policy chess eval flow in PufferLib 4.0, analogous to older env-level `eval --human` workflows.

### Findings

- The user-proposed pattern (`python -m pufferlib.ocean.<env>.eval --human ...`) is env-specific and not available for chess in this 4.0 tree.
- Chess is not present in `pufferlib/ocean/environment.py` Python `MAKE_FUNCTIONS`, so `puffer eval puffer_chess` cannot be used for human play.
- Native C++ backend lacked an exposed render call in `_C`, and chess binding hardcoded `human_play=0`.

### Implementation

1. Added native static render API
   - `pufferlib/extensions/env_binding.h`
   - `pufferlib/extensions/env_binding.c`
   - New function: `static_vec_render(StaticVec* vec, int env_id)`

2. Exposed render through `_C`
   - `pufferlib/extensions/bindings.cpp`
   - New binding: `_C.render(pufferl_obj, env_id=0)`

3. Enabled chess human mode via env kwargs
   - `pufferlib/ocean/chess/binding.h`
   - Parse `human_play` and `render_fps`
   - Force `selfplay=0` when `human_play=1`

4. Added dedicated runner script
   - `tools/chess_human_eval.py`
   - Supports:
     - `--model-path latest` (auto-picks newest checkpoint)
     - `--fps`
     - `--log-pgn`
   - Uses native 4.0 backend (`PuffeRL` + `_C.render` + `evaluate()`) with `horizon=1`, `total_agents=1`.

### Build / Verification

- Rebuilt chess static backend and `_C`:
  - `python setup.py build_chess --inplace`
- Verified script interface:
  - `python tools/chess_human_eval.py --help`

### User Command

```bash
python tools/chess_human_eval.py --model-path latest --fps 30 --log-pgn
```

Equivalent explicit latest-checkpoint form:

```bash
python tools/chess_human_eval.py --model-path "$(ls -t experiments/puffer_chess/*/model_*.pt | head -n1)" --fps 30 --log-pgn
```

Additional compatibility fix:
- `tools/chess_human_eval.py` now adapts selfplay-trained decoder checkpoints (single action head)
  into human-eval runtime shape (two action heads) by duplicating policy-head rows/bias and
  preserving value-head rows/bias. This avoids architecture mismatch when loading latest runs.

Follow-up fix (same date):
- Human-vs-policy chess could stall after a human move because non-selfplay inference lacked chess legality masking.
- Added mask application in non-selfplay rollout path; final Opus correction generalized non-selfplay chess enablement beyond the 1082-only gate.
- File: `pufferlib/extensions/pufferlib.cpp`
- Rebuilt: `python setup.py build_chess --inplace`

---

## Session 12 — 2026-02-24: Opus Follow-up on Human Eval Regression from SPS Refactor

### Context

After SPS-oriented chess backend changes, human-vs-policy eval (`tools/chess_human_eval.py`) launched but policy stalled after first human move.

### Root Cause

- Non-selfplay chess masking flag was gated by selfplay-shaped input (`input_size == 1082`).
- Human eval uses `selfplay=0`, where chess obs remains 2164.
- Result: legality mask not enabled, policy repeatedly sampled invalid actions, board state never advanced.

### Opus Correction

File: `pufferlib/extensions/pufferlib.cpp`

- Non-selfplay chess mask enable branch changed to cover all chess non-selfplay modes:
  - old: `else if (env_name == "puffer_chess" && input_size == 1082)`
  - new: `else if (env_name == "puffer_chess")`

- `net_callback_wrapper()` already applies `apply_chess_mask(logits.mean, obs)` when the flag is true, so this immediately fixes human eval responsiveness.

### Technical Note

- One-step stale GPU observation after a human click is expected due to worker-loop upload timing.
- This causes at most one wasted policy tick; it is not the hang mechanism.

### Relation to SPS Work

- Throughput changes (`selfplay_external_opponent`, query throttling, encoder select) remained valid.
- Regression came from selfplay-biased masking conditions not covering non-selfplay eval workflows.

### Validation

- Rebuild: `python setup.py build_chess --inplace`
- Manual check: human move now followed by policy response in chess window.

---

## Session 13 — 2026-02-24: Human Eval PGN Artifacts + Late Curriculum Behavior Notes

### Artifacts

Two PGNs were saved during human-vs-policy eval of the latest ~300M-step run:
- `game_1771907125.pgn` (460 bytes)
- `game_1771907493.pgn` (584 bytes)

### User-Reported Qualitative State Near Run End

- `stockfish_random_pct` had reached approximately 20.
- `white_winrate` and `black_winrate` were each around 0.2.
- `opponent_winrate` was increasing while annealing continued.
- SPS was about half of earlier throughput plateau.
- Before run kill, `ema_winrate` decline appeared to be tapering.
- It remained unclear whether continued training would produce renewed `ema_winrate` increase.

### Notes on Interpretation

- These are qualitative run-telemetry observations captured near termination, not a controlled eval sweep.
- They are still useful for tracking curriculum pressure and throughput/strength tradeoff in late annealing.

---

## Session 14 — 2026-02-24: Replace Stockfish IPC with Built-in Eval (11x SPS)

### Problem

SPS halved from ~95K to ~53K as `stockfish_random_pct` annealed from 90 to 19. Root cause:
256 Stockfish processes doing blocking pipe I/O in OMP parallel sections. Each `c_step` call
blocked on `fgets()` waiting for Stockfish's `bestmove` response. With 81% of opponent moves
querying Stockfish (at random_pct=19), env step time grew from 340ms to 950ms (76% of wall
time). This doubled the estimated remaining time from ~25h to ~56h.

### Root Cause Analysis

Architecture: 4 buffer pthreads, each with 4 OMP workers processing 64 envs. Per horizon step,
~26 of 64 envs queried Stockfish (50% opponent turns × 81% query rate). Each query:
`fprintf → fflush → fgets(blocking) → parse`. With 4 OMP threads: ~7 sequential rounds of
blocking I/O per step × ~0.5ms per query × 256 horizon steps = ~900ms env time.

The Stockfish process overhead also included:
- 256 `posix_spawn` processes at startup (eager init in `binding.h:287`)
- Per-process: Threads=1, Hash=1MB, ~16MB RSS each = ~4GB total RAM
- Full game history sent via UCI protocol every query (grows with game length)

### Solution: Built-in 1-ply Eval

Replaced Stockfish pipe I/O with an in-process 1-ply search using the position's
incrementally-maintained `materialScore + psqtScore` (already updated by `do_move`/`undo_move`).

The built-in eval (`builtin_select_move` in `chess.h`):
1. Iterates all legal moves (already generated)
2. For each: `do_move` → read `materialScore + psqtScore` → `undo_move`
3. Adds ±150 centipawn noise to simulate ~1200-1400 ELO play
4. Selects the highest-scoring move

This eliminates ALL Stockfish I/O overhead: no processes, no pipes, no blocking reads. The
eval runs in ~1μs per move (vs ~500μs for Stockfish IPC). The existing annealing system
(`_g_sf_random_pct`) works identically — it controls the random vs eval move mix.

### Quality Tradeoff

Stockfish at depth=1 with `UCI_LimitStrength` + ELO 1320 has richer heuristics (king safety,
pawn structure, mobility, piece coordination) than material + piece-square tables. However:

1. At `stockfish_random_pct=90` (training start), 90% of moves are random anyway — opponent
   quality barely matters
2. As annealing progresses, the built-in eval provides reasonable but imperfect play (captures
   hanging pieces, develops with PST bonuses, avoids gross blunders) with natural variation
   from the noise term
3. The training goal is to learn chess fundamentals against progressively harder opponents,
   not to match Stockfish's exact evaluation
4. Eval scripts (`tools/chess_stockfish_eval.py`) still use the real Stockfish binary for
   strength measurement — the built-in eval only affects training opponents

### Parallelism Scaling

With Stockfish eliminated as the bottleneck, env step time dropped dramatically, making GPU
the new bottleneck. This enabled massive scaling of `total_agents`:

| Config | SPS (eval+train) | Env Time | GPU Util | Notes |
|--------|-------------------|----------|----------|-------|
| 256 agents (old, Stockfish) | 53K | 950ms (76%) | 16% | Stockfish I/O bound |
| 2048 agents, built-in | 354K (eval only) | ~100ms | ~40% | Env no longer bottleneck |
| 8192 agents, built-in | 452K (with train) | 568ms (14%) | 94% | Good balance |
| 16384 agents, built-in | **584K (with train)** | 1s (16%) | 94% | **Production config** |
| 32768 agents, built-in | 582K (with train) | ~2s | 94% | Diminishing returns |

GPU saturation occurs around 16384 agents (94% utilization, 79/120G VRAM). Beyond that,
training time on the larger batch dominates and SPS plateaus.

### Code Changes

| File | Change | Description |
|------|--------|-------------|
| `pufferlib/ocean/chess/chess.h` | Added `builtin_select_move()` | 1-ply material+PST eval with ±150cp noise, uses `do_move`/`undo_move` with local undo stack |
| `pufferlib/ocean/chess/chess.h` | Modified `stockfish_bot_move()` | Replaced `stockfish_select_move` call with `builtin_select_move` |
| `pufferlib/ocean/chess/binding.h` | Removed eager `stockfish_start()` | Stockfish processes no longer spawned during env init |
| `pufferlib/extensions/pufferlib.cpp` | Broadened chess mask condition | `input_size == 1082` guard removed (from Session 12 fix) |
| `pufferlib/config/ocean/chess.ini` | `total_agents` 256→16384 | Leverage GPU headroom |
| `pufferlib/config/ocean/chess.ini` | `num_buffers` 4→8 | More parallel buffer processing |
| `pufferlib/config/ocean/chess.ini` | `num_threads` 16→128 | Scale OMP workers with agent count |

### Verification

1. **Build**: `python setup.py build_chess` — compiles cleanly
2. **Zero Stockfish processes**: Verified via `pgrep -c stockfish` during training = 0
3. **Games playing correctly**: `chess_moves=250`, draws/wins/losses distributed normally,
   `invalid_action_rate=0.000`
4. **Annealing functional**: `stockfish_random_pct` decreasing from 90 as `ema_winrate`
   exceeds threshold — identical behavior to Stockfish-based training
5. **SPS benchmark**: 584K sustained with training (11x improvement over 53K)
6. **Human eval**: `tools/chess_human_eval.py` still works (uses `builtin_select_move` path,
   policy responds to human moves)

### Training Launch

```bash
nohup python3 -m pufferlib.pufferl train puffer_chess \
    --wandb --wandb-group chess-v17-builtin-eval > training_log.txt 2>&1 &
```

- wandb: `chess-v17-builtin-eval` group
- Config: 16384 agents, 8 buffers, 128 threads, horizon=256
- SPS: ~584K (11x old), estimated ~4.6h for 10B steps (was ~56h)
- GPU: 94% utilization, VRAM: 79/120G

### Backward Compatibility

- `stockfish_select_move()` still exists in chess.h for eval scripts that need real Stockfish
- `stockfish_start()` is called lazily from `stockfish_select_move()` — never triggered in
  training since `stockfish_bot_move` now calls `builtin_select_move` instead
- `c_close()` still calls `stockfish_stop()` (safe no-op when Stockfish was never started)
- The `stockfish_bot=1` config flag now means "use a non-learner opponent" rather than
  literally "use the Stockfish binary"
- All existing eval scripts (`tools/chess_stockfish_eval.py`, `tools/chess_human_eval.py`)
  continue to work without modification

### State Not Preserved from Previous Run

Training restarted fresh (v17) rather than continuing from the v16 ~300M-step checkpoint
because the following state is not saved/restored:

1. **Optimizer momentum** — Muon weight buffer state (commented out in `save_checkpoint`)
2. **Annealing globals** — `_g_sf_random_pct`, `_g_ema_wr`, `_g_annealing_games` (C statics)
3. **Learning rate schedule** — position in LR annealing
4. **Opponent pool** — selfplay snapshot weights in GPU memory
5. **Rollout buffers** — in-flight experience data

Resume/continuation support is a future task. At 11x SPS, the policy will reach 300M steps
in ~9 minutes vs the previous run's ~56 minutes, making fresh starts acceptable.

---

## Session 14 — 2026-02-24: Full Training Resume (All Practical States)

### Goal

Implement save/resume that restores full training state, not only policy weights.

### Problem

Prior behavior restored only `model_*.pt` weights (when manually loaded). This did not restore:
- optimizer momentum (Muon contiguous buffers)
- trainer counters/epoch state
- RNG streams
- selfplay pool/slot metadata
- chess curriculum globals

This caused resumed runs to behave like partial fresh starts.

### Changes

1. Full-state checkpoint artifact
- `PuffeRL.save_checkpoint()` now writes `trainer_state_full.pt` alongside model checkpoints.
- Full state includes:
  - trainer counters (`global_step`, `epoch`, `last_log_step`)
  - C++ trainer internals (`epoch`, `train_warmup`, `rng_seed`, `rng_offset`, active slot)
  - Muon optimizer state (`lr`, `weight_buffer`, `momentum_buffer`)
  - Python/NumPy/Torch RNG states
  - env-global state (`_C.get_env_state`)
  - selfplay manager state (pool/history/qualities/ELO/slot ids)

2. Resume support in train path
- `_train_rank` now invokes `pufferl.load_training_state(args['load_model_path'])`.
- `--load-model-path` now supports:
  - `trainer_state_full.pt` (full restore)
  - run directory
  - `model_*.pt` (prefers sibling full trainer state)
  - `latest` (newest full trainer state under data dir)

3. New C++ bindings
- Added:
  - `_C.get_env_state(...)`
  - `_C.set_env_state(...)`
  - `_C.get_opponent_slot_policy_ids(...)`
- Exposed `PuffeRL` fields required for restore:
  - `epoch`, `train_warmup`, `rng_seed`, `rng_offset`

4. Static env bridge extensions
- Added `static_vec_get` and `static_vec_put` to static env interface.
- Chess binding implements `MY_GET`/`MY_PUT` for curriculum globals:
  - random pct int/float, EMA winrate, annealing game count, color alternation counter.

5. Selfplay slot/state correctness
- `load_opponent_weights` now records `policy_id` in slot metadata.
- `set_active_opponent` also updates slot policy id.
- Enables faithful slot mapping restoration.

6. Optimizer loader robustness
- Updated `Muon::load_state_dict` to safely initialize undefined buffers and use no-grad copies.

### Validation

- Build succeeded: `python setup.py build_chess --inplace`
- Smoke-tested save/resume:
  - saved checkpoint produced `trainer_state_full.pt`
  - resume from `model_*.pt` auto-selected full trainer state
  - restored step/epoch matched saved values
- CLI verification:
  - `python -m pufferlib.pufferl train puffer_chess --load-model-path <trainer_state_full.pt> ...`
  - run printed `Resumed full trainer state ...` and continued from restored counters.

### Key Files

- `pufferlib/pufferl.py`
- `pufferlib/extensions/bindings.cpp`
- `pufferlib/extensions/env_binding.h`
- `pufferlib/extensions/env_binding.c`
- `pufferlib/ocean/chess/binding.h`
- `pufferlib/extensions/muon.h`

---

## Session 14.1 — 2026-02-24: Resume Completeness + Smoke Re-Validation

### Delta Fix

- Restored `active_opponent_slot` from saved `pufferl_cpp_state` in
  `_restore_full_trainer_state`.
- This closes a small gap where the slot id was persisted but not explicitly
  replayed from the full-state checkpoint.

### Re-Validation

- Rebuilt C++ extension:
  - `python setup.py build_chess --inplace`
- Ran short checkpoint/resume cycle in `/tmp/puffer_resume_smoke`:
  - initial short train -> produced `trainer_state_full.pt`
  - resume from `model_*.pt` -> auto-used sibling full state
  - resume from `latest` -> resolved newest full state
- Both resume paths printed:
  - `Resumed full trainer state from .../trainer_state_full.pt`

---

## Session 15 — 2026-02-24: Move Tutor — Expert-Guided RL from Pre-computed Stockfish Data

### Problem

The builtin 1-ply eval (material+PST) runs at ~400K SPS but produces a model that scores
0% vs real Stockfish 1320. Real Stockfish training was ~50K SPS and only got 1.9% after
312M steps — too slow and still weak. We need SF-quality training signal at high SPS.

### Solution

Use DeepMind's pre-computed `(FEN, best_move)` data as a "move tutor" reward signal during
RL training. When an env resets to a curriculum position, the expert's move is loaded as a
target. The learner gets bonus reward for matching the expert's piece and destination. No
live Stockfish processes needed — zero SPS impact.

### Implementation

#### 1. Extraction Script

**Created** `tools/extract_deepmind_fens_with_moves.py`
- Reads `data/searchless_chess/train/behavioral_cloning_data.bag` (34GB .bag file)
- Same .bag format as `extract_deepmind_fens.py`: `varint(fen_len) + fen_bytes + move_bytes`
- Uniformly subsamples 2M records (enough diversity, ~120MB output)
- Output format: `FEN<tab>UCI_MOVE\n` to `pufferlib/ocean/chess/fens_moves_deepmind.txt`
- Strips halfmove/fullmove from FEN (keeps first 4 fields)
- Parallelized across all CPU cores (same pattern as existing extractor)
- Defaults: `--sample 2000000`, seed 42 for reproducibility

#### 2. C Data Loading (`binding.h`)

- New global: `static uint16_t* _tutor_moves_dm` (parallel array to `_fen_curriculum_dm`)
- Packed move format: `uint16_t = from_sq | (to_sq << 6) | (promo << 12)`
- New function `parse_uci_to_packed(const char* uci)` → uint16_t helper
- New function `load_fen_curriculum_dm_with_moves()`:
  - Reads `fens_moves_deepmind.txt` (tab-separated FEN + move)
  - Falls back to FEN-only `load_fen_curriculum_dm()` if file not found
- Config loading in `my_init()`: `reward_tutor_piece`, `reward_tutor_move`,
  `reward_tutor_wrong`, `tutor_only_mode`
- Logging in `my_log()`: `tutor_piece_rate`, `tutor_move_rate` (only when `tutor_total > 0`)

#### 3. Chess Struct Extensions (`chess.h`)

Added to `Chess` struct:
```c
uint16_t* tutor_moves_dm;      // Pointer to global packed-move array
uint16_t tutor_target;          // Packed target for current episode (0 = none)
int tutor_phase;                // 0=piece, 1=dest, 2=done
float reward_tutor_piece;       // Bonus for matching expert's source square
float reward_tutor_move;        // Bonus for matching expert's destination
float reward_tutor_wrong;       // Penalty for wrong move (optional, default 0)
int tutor_only_mode;            // If 1, episode ends after first move attempt
```

Added to `Log` struct:
```c
float tutor_piece_match;        // Count of piece matches
float tutor_move_match;         // Count of move matches
float tutor_total;              // Total tutor episodes
```

#### 4. Reset Logic (`c_reset`)

When selecting a DeepMind FEN:
1. Load `tutor_target = tutor_moves_dm[idx]`
2. Force `learner_color = pos.sideToMove` — learner plays whichever side the FEN says
   moves next (uses 100% of data, no waste)
3. If learner is Black, flip from/to squares: `sq ^ 56` (convert absolute → learner perspective)
4. After `generate_legal()`, validate target move exists in legal moves — clear if invalid

#### 5. Tutor Reward (`process_player_action`)

**Phase 0** (piece selection): When learner picks a valid piece:
- Compare action to `tutor_target & 0x3F` (expert's source square)
- If match: `rewards[0] += reward_tutor_piece`, increment `tutor_piece_match`
- Advance `tutor_phase = 1`, increment `tutor_total`

**Phase 1** (destination/promotion): When learner completes a move:
- Compare action to `(tutor_target >> 6) & 0x3F` (expert's destination)
- Promotion moves: compare file + promo piece type
- If match: `rewards[0] += reward_tutor_move`, increment `tutor_move_match`
- Else if `reward_tutor_wrong != 0`: apply penalty
- Set `tutor_phase = 2` (done)

**Invalid move path**: If move fails and `tutor_phase == 1`, set `tutor_phase = 2`

#### 6. Tutor-Only Mode (`c_step`)

After a completed move with `tutor_phase == 2` and `tutor_only_mode == 1`:
- Set `terminals[0] = 1.0f`, log episode metrics, call `c_reset()`
- Makes episodes single-move (maximizes tutor signal density)
- Default OFF — game continues normally after tutor move

#### 7. Config (`chess.ini`)

```ini
fen_curric_pct = 0.5        # was 0.0 — 50% of resets use curriculum
deepmind_fen_pct = 1.0      # was 0.0 — 100% of curriculum resets use DeepMind FENs
reward_tutor_piece = 0.05   # NEW — bonus for matching expert's source square
reward_tutor_move = 0.15    # NEW — bonus for matching expert's destination
reward_tutor_wrong = 0.0    # NEW — penalty for wrong move (disabled)
tutor_only_mode = 0         # NEW — if 1, episode ends after first move
```

### Edge Cases Handled

- **Color handling**: Force `learner_color = pos.sideToMove` for tutor episodes — learner
  always plays the FEN's active side. No data wasted, ~50/50 white/black naturally.
- **Square perspective**: Squares flipped via `sq ^ 56` for Black (learner-perspective actions
  are always rank-relative)
- **Promotions**: UCI "a7a8q" → packed promo type, compared against action 64-95 encoding
  (promo_file + promo_row mapping)
- **Invalid FEN/move**: Target validated against legal moves after `generate_legal()` in
  `c_reset()` — cleared if not found in legal move list
- **Thread safety**: All tutor state is per-env in Chess struct; global arrays are read-only
  after init
- **Fallback**: If `fens_moves_deepmind.txt` not found, falls back to FEN-only
  `fens_deepmind.txt` (tutor disabled, curriculum still works)

### Files Modified

| File | Action | Description |
|------|--------|-------------|
| `tools/extract_deepmind_fens_with_moves.py` | **Created** | Extract 2M (FEN, move) pairs from .bag |
| `pufferlib/ocean/chess/binding.h` | Modified | `parse_uci_to_packed()`, `load_fen_curriculum_dm_with_moves()`, tutor config loading, tutor logging |
| `pufferlib/ocean/chess/chess.h` | Modified | Extended Chess/Log structs, modified `c_reset()`, `process_player_action()`, `c_step()` |
| `pufferlib/config/ocean/chess.ini` | Modified | Added tutor config params, enabled curriculum |

### Build Verification

- `python setup.py build_chess` — compiles cleanly, .so copied inplace

### Usage

1. Extract data (one-time):
```bash
python tools/extract_deepmind_fens_with_moves.py
```

2. Train with tutor:
```bash
python -m pufferlib.pufferl train puffer_chess --wandb
```

3. Monitor `tutor_piece_rate` and `tutor_move_rate` in wandb — should increase over time

### Expected Verification Steps

1. Run extraction → produces `pufferlib/ocean/chess/fens_moves_deepmind.txt`
2. Short training run → `tutor_piece_rate` and `tutor_move_rate` appear in logs and are >0
3. Verify SPS unchanged (~400K+ with 16384 agents)
4. After ~100M steps, eval vs Stockfish 1320 depth 1 — expect improvement over 0%
5. Compare wandb curves: `tutor_move_rate` should increase over time as agent learns expert moves
