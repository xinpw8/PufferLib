# DEBUG LOG - Chess Selfplay ELO Convergence Validation

## Session Start: 2026-02-21

### ELO System Bug Fixes (5 iterations)

#### Bug 1: SyntaxError in `update_from_rollout` (pufferl.py:165)
- **Code**: `win_rate = wins / total if total` (no `else` clause)
- **Fix**: `score = wins / total if total > 0 else 0.5`
- **Also**: Line 174 used undefined `score` variable -> now consistent

#### Bug 2: K-factor scaling (pufferl.py:174)
- **Code**: `delta = k * total * (score - expected)` with k=4, total=~2000
- **Problem**: K_eff = 8000, causing wild ELO oscillations (±1000 per epoch)
- **Fix**: `delta = k * (score - expected)` with k=32 (standard ELO per-game K)

#### Bug 3: ELO floor clamping (pufferl.py:178)
- **Code**: `np.maximum(elos, 0.0, out=elos)` with initial ELO=0
- **Problem**: All ELOs clamped to 0, no differentiation possible
- **Fix**: Start at ELO=1000, remove clamping

#### Bug 4: Draw reward confounding ELO (chess.ini)
- **Problem**: `reward_draw=-1.0` makes draws indistinguishable from losses
  in the reward signal. ELO system counted 72% draws as 72% losses.
  Score = wins/total = 14%, expected = 50%, massive ELO crash.
- **Fix**: `reward_draw=0.0` so wins>0, draws≈0, losses<0

#### Bug 5: Quality-weighted sampling bias (pufferl.py:sample_opponent)
- **Problem**: Only recent (strong) opponents selected. Learner never faces
  old (weak) opponents, so ELO never increases relative to baseline.
- **Fix**: 30% epsilon-greedy random opponent selection

### Training Run Log

#### v1-v4: Debugging ELO system
- v1: ELO swung wildly (K-factor bug)
- v2: ELO stayed at 0 (floor clamping bug)
- v3: ELO dropped from 1000 to 140 (draw=loss bug)
- v4: ELO dropped to -200 (same bug, different manifestation)

#### v5: First working ELO (reward_draw=0)
- ELO hovered at 1000±5 (selfplay equilibrium)
- Converged trivially at epoch 200 (baseline=1000, target=1000)
- Stockfish 1320 eval: 0% win rate, 100% losses

#### v6: Epsilon-greedy opponents
- ELO stable at 975-1000
- Policy improving (draw_rate 0.80→0.36, chess_moves 254→165)
- Selfplay ELO doesn't capture absolute improvement

#### v7: 5000 epochs with periodic Stockfish eval (FINAL RUN)
- Epoch 1000 (1.05B steps): Stockfish 1320 = 0% win rate
- Epoch 2000 (2.10B steps): Stockfish 1320 = 0% win rate
- Epoch 3000 (3.15B steps): Stockfish 1320 = 0% win rate
- Epoch 4000 (4.19B steps): Stockfish 1320 = 0% win rate
- Epoch 5000 (5.24B steps): Stockfish 1320 = 0% win rate
- Epoch 5000 (5.24B steps): Stockfish 1320 movetime=1ms = 0% win rate
- Total training time: 1.1 hours
- Final selfplay ELO: 1001.1 (baseline=1000)
- Checkpoint: experiments/puffer_chess/o3acdp76/model_puffer_chess_005000.pt

### Key Observations

1. **Selfplay ELO is near-meaningless for tracking absolute improvement**:
   All opponents come from the same training run. As the policy improves,
   so do all snapshots. Relative performance stays ~50%, ELO stays ~1000.

2. **Policy IS improving by other metrics**:
   - Draw rate: 0.80 → 0.30 (more decisive play)
   - Chess moves: 254 → 170 (shorter, more efficient games)
   - Entropy: 0.81 → 0.43 (more focused action selection)
   - White+Black winrate: 0.13 → 0.35 (more decisive outcomes)

3. **Architecture limitation**: Linear(1082→256) + LSTM(256) + Linear(256→97)
   has ~499K params. AlphaZero used 80M+ params with deep residual CNNs.
   This network cannot learn sufficient chess positional evaluation
   to beat Stockfish 1320. The flat linear encoding destroys spatial
   relationships critical for chess.

### wandb Runs

All runs logged to wandb project `puffer4`, group `chess-selfplay`:
- `elo-convergence` (v1, K-factor bug)
- `elo-convergence-v2` (floor clamping bug)
- `elo-convergence-v3` (draw=loss bug)
- `elo-convergence-v4` (draw=loss still)
- `elo-convergence-v5` (trivial convergence)
- `elo-convergence-v6` (epsilon-greedy)
- `elo-v7-with-stockfish` (final 5000-epoch run)
- `stockfish-baseline` (first Stockfish eval)

### Final Verdict

The selfplay system works mechanically (opponent pool, weight loading,
ELO tracking, quality sampling). The 5 ELO bugs found and fixed are
real improvements to the codebase. But the 499K-param linear network
cannot learn competitive chess from selfplay alone. This confirms the
plan's risk assessment: "Flat Linear+LSTM may plateau before ELO 1000."

Next step: chess-specific CNN encoder with 2D spatial board representation.

---

## Chess Policy Network Architecture Reference

Three chess policy networks exist across PufferLib branches. "ChessOne" was the
working name for the Default flat-linear baseline (never a named class). ChessTwo
and ChessSeven are the two named architectures. ChessNNUE is a third experimental
variant. All share the same 97-action two-phase pick-piece/pick-destination interface.

### Network Comparison

| | ChessOne (Default) | ChessTwo | ChessSeven | ChessNNUE |
|---|---|---|---|---|
| **Location** | 4.0 `pufferlib/models.py` (Default class) | 3.0 `ocean/torch.py` | 3.0 + 4.0 `ocean/torch.py` | 3.0 `ocean/torch.py` |
| **Spatial encoder** | None (flat linear) | 4-layer residual CNN (3x3 convs) | Conv1x1 + channel proj + depthwise 3x3 | NNUE relational token embedding |
| **CNN channels** | — | 256 | 64 (square_dim) | — |
| **Hidden size** | 256 | 512 | 256 | 512 |
| **Embed dim** | — | 32 | 64 (tuned) | 32 (meta) + 256 (rel) |
| **Params (approx)** | ~499K | ~8.5M | ~749K (embed=64) | ~12.6M |
| **Spatial input** | Flat 1082-byte vector | 16ch 8x8 (12 piece + sel + vp + vd + promos_pad) | 3.0: 21ch (17 sq_feat + 4 geo); 4.0: 19ch (15 + 4 geo) | 64 relational tokens from 49K vocab |
| **Residual connections** | No | Yes (1 block: conv→relu→conv + skip) | No (but has spatial_mix residual add) | No |
| **Scalar processing** | Included in flat input | Separate 2-layer MLP (5→512→512) | Concatenated raw (5 values / 255.0) | Separate 2-layer MLP (5→512→512) |
| **Embeddings** | None | side(2,32) castle(16,32) ep(65,32) phase(2,32) | side(2,32) castle(16,64) ep(65,64) phase(2,32) | side(2,32) castle(16,32) ep(65,32) phase(2,32) |
| **Action masking** | No | Yes (from stored obs) | Yes (inline during encode) | Yes (from stored obs) |
| **LSTM wrapper** | Yes (LSTMWrapper) | Compatible | Compatible | Compatible |

### Architecture Details

#### ChessOne (Default / flat baseline)
- `pufferlib/models.py` class `Default`
- `nn.Linear(obs_size, 256) → GELU → LSTM(256,256) → Linear(256, 97)`
- No spatial awareness — flattens the entire 1082-byte observation
- ~499K params. Proven insufficient for chess (can't beat Stockfish at any level)
- Historically the first network tested; confirmed the need for spatial encoders

#### ChessTwo (residual CNN)
- `pufferlib/ocean/torch.py` line 307 (3.0 codebase)
- Spatial path: `Conv2d(16,256,3) → ReLU → Conv2d(256,256,3) → ReLU → Conv2d(256,256,3) + skip → ReLU → Conv2d(256,512,3) → ReLU → flatten`
- Input: 16 channels on 8x8 = 12 piece bitplanes + selected + valid_pieces + valid_dests + promos_padded
- Flattened CNN output: 512 × 64 = 32,768 features
- Scalar path: `Linear(5,512) → ReLU → Linear(512,512) → ReLU`
- Total features: 32,768 + 4×32 + 512 = 33,408 → `Linear(33408, 512) → ReLU`
- Actor: `Linear(512, 97)`, Value: `Linear(512, 1)`
- ~8.5M params. The "beefier" network — true AlphaZero-style residual CNN
- Commented-out lines show it was ported between two different obs layouts

#### ChessSeven (lightweight spatial — the production choice)
- `pufferlib/ocean/torch.py` line 27 (both 3.0 and 4.0)
- Geometric buffer: 4 static planes (diagonal, anti-diagonal, center distance, square color)
- Spatial path: `Conv2d(in,64,1) → ReLU → Conv2d(64,8,1) → ReLU → DepthwiseConv2d(8,8,3,pad=1) + residual → flatten`
- **3.0 input**: 17 per-square features (12 pieces + selected + valid_from + valid_to + us_control + them_control) + 4 geo = **21 channels**
- **4.0 input**: 12 piece bitplanes + selected + valid_pieces + valid_dests + 4 geo = **19 channels** (dropped attack maps)
- Flattened: 8 × 64 + 32 (promos) = 544 board features
- Total features: 544 + 3×64 + 5 = 741 → `Linear(741, 256) → ReLU`
- Actor: `Linear(256, 97)`, Value: `Linear(256, 1)`
- ~749K params at embed_dim=64. Lightweight but spatially aware
- Chosen for production: best throughput/quality tradeoff at ~212K SPS on DGX Spark
- **3.0 and 4.0 are NOT weight-compatible** due to 21 vs 19 input channels

#### ChessNNUE (relational tokens)
- `pufferlib/ocean/torch.py` line 132 (3.0 only)
- Inspired by Stockfish NNUE architecture
- 49,152-entry embedding table (64 squares × 12 piece types × 64 king squares)
- Input: compressed token pairs (2 bytes each, 64 tokens max)
- Relational features: sum of token embeddings (masked by count)
- ~12.6M params. Largest model but requires specialized obs encoding
- Different obs format than ChessTwo/ChessSeven — not interchangeable

### Observation Format Differences

The obs layout changed between 3.0 and 4.0, breaking weight compatibility:

**3.0 (1129 bytes per side):**
- Bytes 0-1087: 64 squares × 17 features (12 pieces + selected + valid_from + valid_to + us_control + them_control)
- Bytes 1088-1119: valid promotions (4×8)
- Bytes 1120-1128: side, castle, ep, pick_phase, self_check, opp_check, rule50, repetition, pass_valid

**4.0 (1082 bytes per side):**
- Bytes 0-767: 12 piece bitplanes (12×64)
- Bytes 768-852: side(2), castle(16), ep(65), phase(2)
- Bytes 853-1044: selected(64), valid_pieces(64), valid_dests(64)
- Bytes 1045-1076: valid promotions (32)
- Bytes 1077-1081: self_check, opp_check, rule50, repetition, pass_valid

### 3.0 Baseline Training Results (2026-02-22)

Running with ChessSeven (embed_dim=64) + LSTMWrapper + tuned hyperparameters:
- 212K SPS on DGX Spark (GB10)
- 748.8K params
- Selfplay ELO ~879 at 11.6B steps (epoch 2600)
- Stockfish eval (Skill Level 4 ≈ ~1200 ELO): 0/20 wins
- Stockfish eval (UCI ELO 1320): 0/20 wins
- Model plays recognizable openings (d4/c3/Qc2, Slav d5/c6) but blunders material by move 20
- wandb run: `chess-3.0-pr15` group

---

## Session: 2026-02-23 — Training vs Stockfish with ChessTwo

### Bugs Found & Fixed

#### Bug 6: OOM from 1024 Stockfish processes
- **Cause**: `total_agents=1024` + `stockfish_bot=1` spawns 1024 independent Stockfish
  processes via `posix_spawn()` in `chess.h:2990`. Each uses 16 MB hash (default) + ~30 MB
  overhead = ~47 GB total. Combined with PyTorch/CUDA = OOM on 119 GB system.
- **Fix 1**: Added `setoption name Hash value 1` in `chess.h:3086` to reduce per-process hash
  from 16 MB to 1 MB. Saves ~15 GB at 1024 agents.
- **Fix 2**: Reduced `total_agents` from 1024 to 256. Now 256 × ~16 MB = ~4 GB stockfish RAM.
- **Result**: 37 GB RAM usage, stable.

#### Bug 7: policy_name config completely ignored
- **Cause**: `chess.ini` had `policy_name = ChessTwo` but the C++ backend in
  `pufferlib/extensions/ocean.cpp:583-585` hardcodes `ChessEncoder` (the ChessSeven-like
  lightweight encoder, ~749K params) for `puffer_chess`. The `policy_name` field is only
  used by Python's `load_policy()` for evaluation, never for training.
- **Result**: Previous training runs showed "Params: 1.2M" (ChessSeven + MinGRU + decoder)
  instead of the intended 20.6M ChessTwo.
- **Fix**: Implemented `ChessTwoEncoder` in C++ (`ocean.cpp`) matching the Python ChessTwo
  architecture: 4-layer residual CNN (16→256→256→256→512 channels, 3x3 convs with padding),
  separate scalar MLP (5→512→512), categorical embeddings, and large projection layer.
- **Verified**: Training now shows "Params: 20.6M".

#### Bug 8: Stockfish `go movetime 1` still too slow (3K SPS)
- **Cause**: Even `movetime=1` has per-call overhead: pipe I/O, search initialization,
  minimum 1ms wall time per call. With 256 envs each making a stockfish call every other
  step, the env step (61% of time) is dominated by synchronous pipe communication.
- **Fix**: Added `stockfish_depth` config option in `chess.h` and `binding.h`. When
  `stockfish_depth > 0`, uses `go depth N` instead of `go movetime N`. `depth 1` returns
  almost instantly since it only evaluates one ply.
- **Config**: `stockfish_movetime_ms = 0`, `stockfish_depth = 1`
- **Result**: SPS improved from 3K to 11K (3.6x). Still 20x slower than pure selfplay (212K)
  but usable — 252h estimated for 10B steps.

### Current Training Run: v8 (ChessTwo vs Stockfish ELO 800)

**Config** (`pufferlib/config/ocean/chess.ini`):
- `policy_name = ChessTwo` (C++ ChessTwoEncoder, 20.6M params)
- `total_agents = 256`, `num_buffers = 4`
- `selfplay = 1`, `stockfish_bot = 1` (stockfish handles opponent moves)
- `stockfish_elo = 800`, `stockfish_depth = 1`
- `horizon = 256`, tuned PPO hyperparams

**Early metrics (epoch 24, 1.6M steps, ~3 min):**
- SPS: 11.1K
- VRAM: 113.9/120 GB (tight but stable)
- RAM: 53 GB / 119 GB
- white_winrate: 0.0, black_winrate: 0.0 (losing every game)
- material_score: -17 (losing massive material)
- chess_moves: 36 (games ending quickly)
- elo: 861 (selfplay ELO, dropping since losing to stockfish)

**Launch command**:
```bash
source .venv/bin/activate
PUFFER_STOCKFISH_PATH=/usr/games/stockfish python3 -m pufferlib.pufferl train puffer_chess --wandb --wandb-group chess-v8-stockfish
```

**wandb**: https://wandb.ai/xinpw8/puffer4/runs/oqypw0nd (run: `daily-energy-59`)
**Log**: `/tmp/chess_4.0_v2_training.log`
**PID**: 23566

### Architecture: C++ ChessTwoEncoder

Located in `pufferlib/extensions/ocean.cpp`.

```
Input: 16ch 8x8
  12 piece bitplanes + selected + valid_pieces + valid_dests + promos_padded
  ↓
Conv2d(16, 256, 3x3, pad=1) → ReLU
  ↓
Conv2d(256, 256, 3x3, pad=1) → ReLU  ← residual block start
Conv2d(256, 256, 3x3, pad=1) + skip → ReLU
  ↓
Conv2d(256, 512, 3x3, pad=1) → ReLU → flatten → (B, 32768)
  ↓
Cat with: embeddings(128) + scalar_MLP(512)
  ↓
Linear(33408, 512) → ReLU → MinGRU(512) → Actor(512,97) + Value(512,1)
```

Total params: 20.6M (vs 1.2M for ChessSeven, vs 499K for flat linear)

## Session: 2026-02-23 — Critical Eval Bug, Reward Shaping, Metric Granularity

### Bug 9: Eval script evaluates RANDOM WEIGHTS (bf16 sync missing)

**Severity**: CRITICAL — all previous Stockfish eval results are invalid.

**Root cause**: PufferLib 4.0 with `USE_BF16=true` (the default, compile-time constant in
`models.cpp`) maintains two separate policy models:

- `policy_fp32`: Master weights for the optimizer. Parameters are **views** into
  `muon.weight_buffer` (set via `Tensor::set_data()` in `Muon` constructor).
- `policy_bf16`: Inference weights in bfloat16. A **completely separate** `torch::nn::Module`.
  Used for all forward passes during rollouts (`pufferlib.cpp:352`).

The sync from fp32 → bf16 happens **only** inside `train_impl()` (`pufferlib.cpp:637`):
```cpp
if (USE_BF16) {
    sync_policy_weights(pufferl.policy_bf16, pufferl.policy_fp32);
}
```

The eval script (`tools/chess_stockfish_eval.py`) calls only `evaluate()` (which calls
`_C.rollouts()`), never `_C.train()`. So bf16 weights are never synced.

**`_load_checkpoint_into_muon()`** loaded weights into `muon.weight_buffer`, which correctly
updated `policy_fp32` params (they're views into the buffer). But `policy_bf16` — the model
actually used for inference — retained its **random initialization weights**.

**Evidence**:
```
bf16 is fp32: False
bf16 encoder.conv1.weight: dtype=torch.bfloat16, data_ptr=13673146880
fp32 encoder.conv1.weight: dtype=torch.float32, data_ptr=277720638423040
Same storage: True  ← fp32 params ARE views into weight_buffer, confirmed
```
Two different objects, two different storage addresses. Writing to weight_buffer updates fp32
but not bf16.

**Impact**: Every Stockfish eval run since the eval script was created (Session 4, 2026-02-21)
evaluated random bf16 weights. This includes:

| Run | Reported Result | Actual |
|-----|----------------|--------|
| v5 epoch 200: SF 1320 | 0% win rate | Random weights, meaningless |
| v7 epoch 1000-5000: SF 1320 | 0% win rate × 6 evals | Random weights, meaningless |
| v7 epoch 5000: SF 1320 movetime=1ms | 0% win rate | Random weights, meaningless |
| v8 epoch 3400: SF 800 mt=30ms | 0% win, 0% draw, 100% loss | Random weights, meaningless |
| v8 epoch 3600: SF 800 mt=0 depth=1 | 0% win, 0.66% draw, 99.3% loss | Random weights, meaningless |

**The fix** (`tools/chess_stockfish_eval.py`):
```python
def _load_checkpoint(pufferl_cpp, model_path):
    # 1. Load into muon.weight_buffer (updates policy_fp32 via views)
    state_dict = torch.load(model_path, map_location="cpu")
    wb = pufferl_cpp.muon.weight_buffer
    # ... copy into weight_buffer ...

    # 2. CRITICAL: Sync bf16 from fp32
    bf16, fp32 = pufferl_cpp.policy_bf16, pufferl_cpp.policy_fp32
    if bf16 is not fp32:
        with torch.no_grad():
            for p_bf16, p_fp32 in zip(bf16.parameters(), fp32.parameters()):
                p_bf16.data.copy_(p_fp32.data)

    # 3. Validate the sync
    # Compare first param between checkpoint and bf16 policy
    max_diff = (src - dst).abs().max().item()
    assert max_diff < 0.01, "Sync failed"
```

**Validated result** (after fix):
```
Checkpoint validation OK: bf16 param 'encoder.conv1.weight' max_diff=0.000000
games=100/100 W=0.0 D=67.2 L=32.8 win_rate=0.000 draw_rate=0.672
```
This now matches training metrics: draw_rate=0.658 in training vs 0.672 in eval. ✓

**How this was discovered**: Training reported 65% draw rate but eval (same model, same SF
settings) reported 0.66% draws and 99.3% losses. Initially misattributed to Stockfish strength
difference (eval used movetime=30ms vs training's 0ms). Re-running with identical settings
(movetime=0, depth=1) still showed 0.66% draws. This ruled out Stockfish strength. Comparing
eval-with-model vs eval-without-model produced identical results (~0.7% draws, ~99.3% losses),
proving the model weights weren't being used. Tracing the inference path through pufferlib.cpp
revealed `policy_bf16` (line 352) is used for rollouts, not `policy_fp32`. Confirming
`bf16 is fp32 → False` and different `data_ptr` values proved they're separate objects.

**Note on precision config**: `default.ini` has `precision = float32`. This is a **training
config value** that has no effect on the compiled C++ code. `USE_BF16` is a compile-time
constant (`models.cpp:15`) defaulting to `true` unless `-DPRECISION_FLOAT` is passed during
build. The `setup.py` ProfilerBuildExt respects `--precision=float` but the main `_C` extension
build does not read the config. The training pipeline already uses bf16 for inference and fp32
for the optimizer regardless of the config setting.

---

### Metric Improvements: Draw Type Breakdown + Loss Tracking

**Problem**: `white_winrate=0` and `black_winrate=0` appeared inconsistent because there was no
corresponding loss metric. The 35% non-draw games were losses, but this was only inferrable by
subtraction (1 - draw_rate - win_rate). Also, all draws were lumped together with no way to
distinguish draw mechanisms (repetition vs 50-move vs stalemate vs insufficient material).

**Changes to `chess.h`**:

1. `game_result_with_legal_count()` now returns distinct codes:
   - 0 = game continues
   - 1 = Black wins (White checkmated)
   - 2 = White wins (Black checkmated)
   - 3 = Draw by stalemate
   - 4 = Draw by insufficient material
   - 5 = Draw by 50-move rule
   - 6 = Draw by threefold repetition

2. `end_game()` updated: `game_result == 3` → `game_result >= 3` for draw detection,
   plus per-type tracking.

3. `Log` struct extended with 6 new fields:
   - `white_lossrate`, `black_lossrate` (explicit loss tracking by color)
   - `draw_by_stalemate`, `draw_by_insufficient`, `draw_by_50move`, `draw_by_repetition`

**Changes to `binding.h`**: All 6 new fields exported via `my_log()` → wandb.

---

### Reward Shaping Changes

**Problem**: All reward shaping was disabled (`reward_draw=0.0, reward_material=0.0`, etc.).
The agent received only sparse terminal rewards (+1 win, -1 loss, 0 draw). This created a
local optimum: the agent learned "draws (reward=0) > losses (reward=-1)" and got stuck in a
draw-seeking equilibrium, shuffling pieces until 50-move or repetition rules triggered.

Evidence: 65% draw rate, 0% win rate, material_score=-9.4 (losing ~9 pawns on average),
entropy=0.03 (policy collapsed to near-deterministic).

**Changes to `chess.ini`**:
```ini
reward_draw = -0.3        # was 0.0 — penalize draw-seeking
reward_material = 0.05    # was 0.0 — gradient for captures/material
reward_check = 0.01       # was 0.0 — encourage attacking king
reward_repetition = -0.05 # was 0.0 — punish piece shuffling
ent_coef = 0.02           # was 0.006 — prevent policy collapse
```

**Rationale**:
- `reward_draw=-0.3`: Draws must be worse than 0 to break the draw-seeking equilibrium.
  Not as bad as losing (-1.0) to still prefer draws over losses.
- `reward_material=0.05`: The existing dense reward code in `c_step()` computes material
  deltas (using SEE for captures) but was scaled to 0. Setting to 0.05 gives a gradient
  toward good trades without overwhelming the terminal signal.
- `reward_check=0.01`: Small bonus for putting opponent in check. Encourages aggressive play.
- `reward_repetition=-0.05`: Penalizes position repetition. Directly combats the shuffling
  strategy.
- `ent_coef=0.02`: Entropy was 0.03 (near collapse). Higher entropy coefficient maintains
  exploration.

---

### Validated Eval Results (FIRST CORRECT EVAL)

After bf16 sync fix, epoch 3600 checkpoint (~240M steps) vs Stockfish ELO 800 depth=1:

| Metric | Training (live) | Eval (100 games) |
|--------|----------------|------------------|
| Win rate | 0% | 0% |
| Draw rate | 65.8% | 67.2% |
| Loss rate | 34.2% | 32.8% |

Training and eval now agree. The model genuinely cannot win against Stockfish ELO 800 at
depth 1 after 240M steps with purely sparse rewards. It draws ~2/3 of games (via repetition
and 50-move rule) and loses the other ~1/3.

**Next steps**: Restart training with reward shaping enabled and new metrics. The C code
changes require recompilation (`python setup.py build_chess` + copy .so).

---

## Bug 10: Stockfish Missing Game History — Threefold Repetition Exploit

**Date**: 2026-02-23
**wandb**: https://wandb.ai/xinpw8/puffer4/runs/wbq70goz (v9, pre-fix)

### Symptom

After adding reward shaping (v9), `draw_by_repetition` climbed to 45%+ of all game outcomes
within 42.5M steps. ALL draws were threefold repetition. The agent learned to force draws
via repetition despite the -0.3 draw penalty and -0.05 repetition penalty, because Stockfish
appeared to cooperate.

### Root Cause

`stockfish_select_move()` in `chess.h` line 3173 sent only the current FEN to Stockfish:
```c
fprintf(env->stockfish_in, "position fen %s\n", fen);
```

**No game history was sent.** Stockfish sees each position as fresh — it cannot detect
threefold repetition because it has no knowledge of prior positions. For any given board
state, Stockfish always returns the same "best" depth-1 move. The agent learns to create
a loop: play move A, Stockfish plays move B, play move C that returns to the original
position, Stockfish plays the same move B again (it does not know it already played it).
After 3 cycles, the env counts threefold repetition → draw.

**Threefold repetition defined**: A draw is declared when the same position (same piece
placement, same side to move, same castling rights, same en passant square) occurs 3 times
during the game. The positions do not need to be consecutive.

**Why Stockfish "cooperated"**: Stockfish was NOT choosing to draw. It literally could not
know it was repeating — each `position fen` command made it think it was analyzing a fresh
game. Its "best move" was correct for that position in isolation, but without history context,
it had no reason to vary its play.

### Fix

Two changes in `chess.h`:

1. **`stockfish_select_move()`**: Send full game history via UCI protocol:
```c
// Before (broken):
fprintf(env->stockfish_in, "position fen %s\n", fen);

// After (fixed):
if (env->pgn_move_count > 0) {
    fprintf(env->stockfish_in, "position fen %s moves", env->starting_fen);
    char uci[8];
    for (int i = 0; i < env->pgn_move_count; i++) {
        move_to_uci(env->pgn_moves[i], uci);
        fprintf(env->stockfish_in, " %s", uci);
    }
    fprintf(env->stockfish_in, "\n");
} else {
    fprintf(env->stockfish_in, "position fen %s\n", env->starting_fen);
}
```

2. **Move recording guards**: `pgn_moves[]` was only populated when `human_play || log_pgn`.
   In training, both are 0, so `pgn_move_count` stayed 0. Added `stockfish_bot` to guards:
```c
// Learner move recording (line 2753):
if ((env->human_play || env->log_pgn || env->stockfish_bot) && ...)

// Opponent move recording (line 3256):
if ((env->log_pgn || env->stockfish_bot) && ...)
```

3. **Added `move_to_uci()` helper**: Converts internal Move to UCI string (e.g., "e2e4",
   "e7e8q" for promotion).

### Validation

v10 training (wandb: https://wandb.ai/xinpw8/puffer4/runs/vngs0qmz):
- At 786K steps (1167 games): `draw_by_repetition = 0.000` (was 45%+ in v9 at same point)
- Agent losing ~100% as expected for untrained policy vs Stockfish with history awareness

---

## Bug 11: Stockfish UCI_Elo minimum is 1320, not 800

**Date**: 2026-02-23

`stockfish_elo = 800` in chess.ini was silently clamped to 1320 (Stockfish's minimum).
All v8-v12 runs trained against Stockfish 1320, not 800. This partially explains the
100% loss rate — SF 1320 at depth 1 is too strong for a randomly initialized network.

**Fix**: Set `stockfish_elo = 1320` (honest) and add `stockfish_random_pct` to weaken it.

## Bug 12: No curriculum — model gets zero positive reward signal

**Date**: 2026-02-23

v12 entropy collapsed from 0.85 → 0.06 within 17M steps. 100% loss rate against SF 1320
means every trajectory has reward ≈ -1.0. No positive signal → no gradient toward winning
play → policy collapses to a fixed bad sequence.

**Fix**: Added `stockfish_random_pct` parameter (`chess.h`, `binding.h`). When stockfish's
turn comes, `(rand() % 100) < stockfish_random_pct` → play a random legal move instead
of stockfish's move. This creates a tunable opponent strength below Stockfish's minimum.

**Config**: `stockfish_random_pct = 90` (90% random, 10% stockfish = very weak opponent)

### v13: Training with 90% random stockfish (wandb group `chess-v13-random90`)

**Launch command**:
```bash
PUFFER_STOCKFISH_PATH=/usr/games/stockfish python3 -m pufferlib.pufferl train puffer_chess --wandb --wandb-group chess-v13-random90
```

**Early metrics (epoch 7, 459K steps, ~1 min):**
- SPS: 22.1K (2x faster — random moves skip stockfish pipe)
- entropy: 0.920 (healthy, no collapse)
- white_winrate: 3.8%, black_winrate: 2.2% (FIRST WINS EVER)
- draw_rate: 70% (expected vs mostly-random opponent)
- material_score: -4.5 (much better than -17)
- chess_moves: 262 (real-length games)

**wandb**: https://wandb.ai/xinpw8/puffer4/runs/rnxk83ag

**Progression (observed from training):**
- Epoch 200 (13M steps): ~80% win rate vs 90% random SF
- Epoch 600 (39M steps): ~93% win rate
- Epoch 1000 (65.5M steps): ~94.6% win rate (training still running)

### v13 Evaluation Results (epoch 1000, 65.5M steps)

**Bug 13: Eval inherited `stockfish_random_pct=90` from chess.ini**

First eval showed 98% win rate, but was actually playing against 90% random stockfish
(same weak opponent as training). Fixed eval script to accept `--stockfish-random-pct`
flag and default to 0 (full strength).

**Strength curve (100-200 games per point):**

| Opponent (`random_pct`) | Win Rate | Draw Rate | Loss Rate |
|--------------------------|----------|-----------|-----------|
| 0% (real SF 1320 d=1) | 0.0% | 1.5% | 98.5% |
| 50% | 57.6% | 17.6% | 24.8% |
| 70% | 81.3% | 12.4% | 6.3% |
| 80% (training-1) | 94.7% | 2.5% | 2.8% |
| 90% (training level) | 98.0% | 2.0% | 0% |
| 95% | 100% | 0% | 0% |

The agent breaks even (~50% WR) at ~50% random. It cannot win a single game against
real Stockfish 1320 depth=1.

### PGN Game Analysis (200 games vs real SF 1320 d=1)

**Bug 14: PGN labels swapped in `export_pgn_append()`**

`chess.h:3763` had `learner_color == CHESS_BLACK ? "Learner" : "Opponent"` for White label,
which is backwards. Fixed to `learner_color == CHESS_WHITE ? "Learner" : "Opponent"`.

**Opening patterns:**
- As White: plays 1.h4 in 100% of games (not a real opening)
- As Black: plays 1...h5 in 82% of games (terrible)
- First 4 moves are ALL pawns in 100% of White games
- Never castles (0 out of 234 games)

**Fatal weaknesses:**
1. H-pawn rush strategy (h4-h5-h6-hxg7) dominates play — works vs random, useless vs SF
2. Zero piece development — plays all-pawn game, knights/bishops developed <10% of games
3. Falls into same Qa5+/Qxb4+ trap in 25% of White games
4. Queen blundered in 36% of all games
5. No king safety — King stays in center, forced to move by move 12 in 62% of White games

**Game length:** Mean 22.5 moves (min 7, max 72). 68% end by move 24.

**The 3 draws:** All from SF entering perpetual check or failing to mate in won endgame.
None from good learner play.

**Diagnosis:** Agent plays at ~100-200 Elo. It learned to push pawns (which works vs random)
but has not learned fundamental chess concepts: develop pieces, control center, castle, avoid
hanging pieces. The 90% random training created a local optimum where pawn-pushing wins.

### Next Steps

The agent needs to learn from stronger signal. Options:
1. **Curriculum training**: Gradually reduce `stockfish_random_pct` from 90→80→70→...
   - Pro: Builds on existing training, natural progression
   - Con: Pawn-pushing strategy may be hard to unlearn
2. **Imitation learning**: Pre-train on expert games (Stockfish evaluations of positions)
   - Pro: Teaches real chess patterns directly
   - Con: Requires supervised training pipeline changes
3. **Reward shaping**: Heavier piece development rewards, castling bonus
   - Pro: Easy to implement, can combine with curriculum
   - Con: May create other degenerate strategies

---

## Session: 2026-02-23 — v15 Throughput Push (>100K SPS), Validity Audit, Resume Semantics

### User Request

Increase chess training throughput from ~22K SPS to >100K SPS without sacrificing the
curriculum behavior.

### Baseline (Before Changes)

Active run (`chess-v14-curriculum`, wandb run `6ovobqug`) showed:
- SPS: ~19K–22K (stable ceiling)
- stockfish_random_pct auto-curriculum descending from 90 -> 40 -> 30
- Env time dominated runtime as random_pct decreased (more Stockfish calls)

### Root Causes Identified

1. **Stockfish call overhead remained synchronous and expensive**
   - Even at depth=1, querying Stockfish every opponent turn kept env step expensive.

2. **Training path remained over-provisioned for throughput target**
   - C++ backend used heavy chess path behavior unless explicitly configurable.

3. **Selfplay callback still computed opponent policy forwards even when unused**
   - With `stockfish_bot=1`, environment executes opponent moves; opponent logits sampled in
     `net_callback_selfplay()` were discarded by env logic.
   - This wasted forward compute every rollout step.

### Code/Config Changes Applied

#### 1) Added explicit Stockfish query throttle (new env knob)

- New config/env field: `stockfish_query_pct` (0-100)
- Behavior: on opponent turn, query Stockfish only with this probability; otherwise play random legal move
- Kept existing `stockfish_random_pct` semantics (applies when querying Stockfish)
- Logged metric added: `stockfish_query_pct`

Files:
- `pufferlib/ocean/chess/chess.h`
- `pufferlib/ocean/chess/binding.h`
- `pufferlib/config/ocean/chess.ini`

#### 2) Made C++ chess encoder selectable for training

- Added `chess_encoder` selection path in C++ policy creation:
  - `1` = `ChessEncoder` (fast)
  - `2` = `ChessTwoEncoder` (heavier)
- Wired value from env kwargs to `create_policy(...)`.

Files:
- `pufferlib/extensions/ocean.cpp`
- `pufferlib/extensions/pufferlib.cpp`

#### 3) Added selfplay fast path for env-controlled opponent

- New flag: `selfplay_external_opponent`
- When `env_name == puffer_chess && stockfish_bot=1`, skip opponent-policy forward/sampling in
  `net_callback_selfplay()` and zero opponent action slots.
- Learner action path remains unchanged.

File:
- `pufferlib/extensions/pufferlib.cpp`

#### 4) Throughput-oriented chess config

`pufferlib/config/ocean/chess.ini`:
- `chess_encoder = 1`
- `hidden_size = 256`
- `stockfish_query_pct = 10`
- `stockfish_random_pct = 90` (curriculum start unchanged)
- `total_agents = 256`, `num_buffers = 4`, `horizon = 256` (unchanged from v14 run setup)

### Throughput Validation

Short probes:

1. After encoder/query knobs only:
- SPS: ~46K–48K (improved but below target)

2. After selfplay external-opponent fast path:
- SPS: ~104K–110K (target achieved)

3. Long run verification (`chess-v15-100k`, wandb run `re8nadd7`, name `desert-fire-68`):
- Sustained SPS: ~103K–122K
- Typical range: 109K–119K
- Stockfish curriculum remained active

### Metric Validity Audit (Important)

At high-performance phase in `desert-fire-68`:
- white_winrate ~0.45
- black_winrate ~0.45
- opponent_winrate ~0.004–0.011
- stockfish_random_pct reached 0.0

This looked like near-parity vs "full Stockfish", but was misleading because:
- `stockfish_query_pct` in training was 10
- So even at `stockfish_random_pct=0`, only ~10% of opponent turns were true Stockfish moves

Meaning: training metrics measured performance vs a mostly-random opponent mixture, not true
"Stockfish every turn" strength.

### Eval Script Validity Fix

Found leakage issue in eval setup:
- `tools/chess_stockfish_eval.py` inherits env config from `chess.ini`
- Without override, eval also used training `stockfish_query_pct=10`

Fix:
- Added CLI arg `--stockfish-query-pct` (default 100)
- Forced `env_cfg["stockfish_query_pct"]` from CLI in eval script
- Added fields to `EvalSummary`

File:
- `tools/chess_stockfish_eval.py`

### Stockfish Sweep (Checkpoint `re8nadd7/model_puffer_chess_001400.pt`)

Settings:
- Stockfish ELO: 1320 (minimum), depth=1, movetime=0
- Games per point: 200
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

Control run (to match training conditions):
- `random_pct=0`, `query_pct=10` -> win_rate 0.959

Conclusion:
- Reported high training winrates were internally consistent for `query_pct=10`
- True strength vs "always-query" Stockfish remained low at low random_pct
- Around random_pct ~16 (user’s noted ~0.16), checkpoint still near 0% wins

Artifacts:
- `data/eval_sweep_001400/pct_*.json`
- `data/eval_sweep_001400/pct_*.log`
- control: `data/eval_sweep_001400/pct_0_query10.json`

### Resume Semantics Investigation (pretty-pyramid-69 vs desert-fire-68)

Question: why did resumed run (`pretty-pyramid-69`, run `kzfg8oi4`) start with fresh-looking
winrates and curriculum?

Root cause:
- `--load-model-path` is not applied in the C++ training path (`_train_rank` constructs
  `PuffeRL` and starts loop directly).
- `load_model_path` handling exists in `load_policy()` (eval path), not in `train(...)` path.
- Curriculum globals (`_g_sf_random_pct`, counters) are process-global C statics and reset on
  new process start.

Observed startup evidence:
- epoch 1 showed `stockfish_random_pct=90`, `elo=1000`, and fresh early rates
- consistent with new-run initialization, not true continuation

Relevant files:
- `pufferlib/pufferl.py` (`_train_rank`, `load_policy`)
- `pufferlib/ocean/chess/chess.h` (curriculum globals)
- `pufferlib/ocean/chess/binding.h` (curriculum initialization)

### Final State at End of Session

- Throughput target met (`>100K SPS`) with stable run behavior.
- Eval pipeline corrected to support valid "full Stockfish query" sweeps.
- Discovered that current "resume" CLI behavior does not resume training state in C++ path.
- Training restarted under `pretty-pyramid-69` (`kzfg8oi4`) from checkpoint path argument, but
  behavior confirmed this is effectively a fresh run under current code.

### Session 11: Human-vs-Policy Chess Eval Path (2026-02-24)

Problem:
- 4.0 had no direct chess equivalent of env-level `python -m ...eval --human`.
- `puffer eval puffer_chess` is not available because chess is not wired into `pufferlib/ocean/environment.py` Python env creators.
- Native C++ training/eval backend had no exposed render hook, and chess binding forced `human_play=0`.

Changes made:
1. Added static render hook in native env interface:
   - `pufferlib/extensions/env_binding.h`: declared `static_vec_render(StaticVec* vec, int env_id)`
   - `pufferlib/extensions/env_binding.c`: implemented it via `c_render(&envs[env_id])`
2. Exposed render to Python C++ bindings:
   - `pufferlib/extensions/bindings.cpp`: added `_C.render(pufferl_obj, env_id=0)`
3. Enabled chess human mode from config kwargs:
   - `pufferlib/ocean/chess/binding.h` now parses:
     - `human_play`
     - `render_fps`
   - If `human_play=1`, forces `selfplay=0`.
4. Added interactive runner:
   - `tools/chess_human_eval.py`
   - Loads latest checkpoint (or explicit `--model-path`), enables human mode, renders window, and steps policy one tick at a time.

Build step:
- `python setup.py build_chess --inplace`

Launch command (latest checkpoint):
- `python tools/chess_human_eval.py --model-path latest --fps 30 --log-pgn`

Notes:
- In-window controls are mouse-based move selection.
- Start screen lets human choose white/black.
- `ESC` exits.

Compatibility note:
- Most recent checkpoints were trained with `selfplay=1` (single action head in decoder).
- Human eval config uses `selfplay=0` for legality (`selfplay && human_play` is invalid in chess env logic).
- `tools/chess_human_eval.py` therefore includes decoder-head adaptation for checkpoint load:
  - duplicates policy logits into both heads
  - keeps value head unchanged

### Session 11.1: Human-vs-policy no-response fix

Symptom:
- In human eval, after human move (e.g., white), AI (black) did not respond.

Root cause:
- Chess legality masking (`apply_chess_mask`) was enabled only in `selfplay=1` path.
- Human eval runs with `selfplay=0` (required by env constraints), so policy sampled unmasked actions.
- In this mode, repeated invalid-action sampling can appear as "AI does nothing".

Initial fix attempt:
- Applied mask in `net_callback_wrapper()` before `sample_actions()`.
- This was later corrected in Session 11.2 to remove the overly strict `input_size == 1082` non-selfplay gate.

File:
- `pufferlib/extensions/pufferlib.cpp`

Build:
- Rebuilt with `python setup.py build_chess --inplace`.

### Session 11.2: Opus Correction — Human Eval Hang Caused by SPS-Path Assumptions

Issue observed by user:
- In `tools/chess_human_eval.py`, human could move once, then policy never responded.

Why this happened:
- During SPS optimization work, chess action masking was wired around selfplay assumptions.
- Selfplay halves chess obs to 1082 per side, so mask enablement logic was initially keyed to that shape.
- Human eval runs with `selfplay=0`, so full obs stays 2164 and the old condition skipped masking.
- Without mask, policy repeatedly sampled invalid actions on unchanged state, appearing "stuck".

Opus correction applied:
- In `pufferlib/extensions/pufferlib.cpp`, non-selfplay chess mask enable condition was broadened:
  - from: `else if (env_name == "puffer_chess" && input_size == 1082)`
  - to:   `else if (env_name == "puffer_chess")`
- Non-selfplay rollout path already masks logits before sampling (`net_callback_wrapper`), so enabling this flag fixed the hang.

Additional note from root-cause analysis:
- `c_render` mutates CPU board/obs immediately on human click, while GPU obs sync happens in worker loop.
- This can cause one stale-inference tick right after a human move (one wasted step), then recovers.
- Not the primary hang cause.

Relation to SPS changes:
- `selfplay_external_opponent` fast path and selfplay-centric masking were correct for throughput.
- Regression was incomplete coverage of non-selfplay chess eval mode introduced during that throughput-focused refactor.

Verification checklist used:
1. Rebuild: `python setup.py build_chess --inplace`
2. Run: `python tools/chess_human_eval.py --model-path latest --fps 30 --log-pgn`
3. Choose color, make move, confirm policy replies promptly.

### Session 12.1: Human-Eval PGNs + Late-Run Qualitative Trend Notes (300M-step run)

Artifacts captured from human-vs-policy eval:
- `game_1771907125.pgn` (460 bytes)
- `game_1771907493.pgn` (584 bytes)

These were saved from interactive human eval against the most recent ~300M-step checkpoint.

User-reported qualitative training-state observations at termination time:
- `stockfish_random_pct` had annealed to approximately 20.
- `white_winrate` and `black_winrate` were both around 0.2.
- `opponent_winrate` was rising as annealing progressed.
- SPS was roughly halved relative to earlier high-throughput phase.
- `ema_winrate` decline appeared to be starting to taper just before run termination.
- Unknown whether `ema_winrate` would stabilize and re-rise from that point without continued training.

Interpretation note:
- These are in-run telemetry trends (not controlled gate eval results).
- They are useful for curriculum dynamics tracking but not sufficient alone for strength claims.

### Session 14: Full Save/Resume for Training State (2026-02-24)

Objective:
- Implement real training resume that restores optimizer/trainer/env state, not just model weights.

What was missing before:
- `train` path ignored true checkpoint restoration semantics.
- `model_*.pt` only restored weights (if manually loaded); optimizer momentum, trainer counters,
  RNG streams, selfplay metadata, and chess curriculum globals were reset.

Implementation summary:

1. Full trainer snapshot file
- Added `trainer_state_full.pt` written at every checkpoint save.
- Saved contents include:
  - model identity + run metadata
  - `global_step`, `epoch`, `last_log_step`
  - C++ trainer state (`pufferl_cpp.epoch`, `train_warmup`, `rng_seed`, `rng_offset`, active slot)
  - Muon optimizer state (`lr`, contiguous `weight_buffer`, `momentum_buffer`)
  - RNG states: Python `random`, NumPy, Torch CPU, Torch CUDA(all devices)
  - env-global state via C bridge (`_C.get_env_state`)
  - selfplay manager state (opponent pool/history/qualities/ELO/swap counters + slot ids)

2. Resume loader wired into train path
- `_train_rank` now calls `pufferl.load_training_state(args['load_model_path'])` when provided.
- `--load-model-path` accepts:
  - `trainer_state_full.pt` (full restore)
  - run directory containing checkpoint files
  - `model_*.pt` (auto-detect sibling `trainer_state_full.pt` and prefer full restore)
  - `latest` (auto-picks newest full trainer state under `data/<env>/*/trainer_state_full.pt`)

3. C++ bindings added for state transfer
- New bindings:
  - `_C.get_env_state(pufferl_cpp)`
  - `_C.set_env_state(pufferl_cpp, state_dict)`
  - `_C.get_opponent_slot_policy_ids(pufferl_cpp)`
- Exposed C++ trainer fields for restore:
  - `pufferl_cpp.epoch`, `train_warmup`, `rng_seed`, `rng_offset`

4. Static env bridge hooks
- Added `static_vec_get` / `static_vec_put` in static env binding layer.
- Chess binding now implements `MY_GET`/`MY_PUT` to persist curriculum globals:
  - `_g_sf_random_pct`, `_g_sf_random_pct_f`, `_g_ema_wr`, `_g_annealing_games`, color counter.

5. Selfplay slot bookkeeping fix
- `_C.load_opponent_weights(..., slot, policy_id)` now sets slot policy id.
- `_C.set_active_opponent(..., slot, policy_id)` now also writes slot policy id.
- Enables faithful slot/active-opponent restoration.

6. Muon load robustness fix
- `Muon::load_state_dict` now handles undefined momentum buffers safely and uses no-grad copy.

Validation done:
- Built: `python setup.py build_chess --inplace`
- Smoke test:
  - save checkpoint -> `trainer_state_full.pt` created
  - resume from `model_*.pt` -> auto-full-restore path used
  - restored `global_step`/`epoch` matched saved state
  - CLI train resume ran and printed `Resumed full trainer state ...`

Usage:
- Resume with full state:
  - `python -m pufferlib.pufferl train puffer_chess --load-model-path /path/to/trainer_state_full.pt`
- You can also pass a model file in same run dir:
  - loader will prefer sibling `trainer_state_full.pt` automatically.

### Session 14.1: Resume Validation Pass + Active Slot Restore Completion (2026-02-24)

Small follow-up fix:
- `_save_full_trainer_state()` already saved `pufferl_cpp_state.active_opponent_slot`.
- `_restore_full_trainer_state()` now restores this field directly:
  - `self.pufferl_cpp.active_opponent_slot = int(cpp_state['active_opponent_slot'])`

Why:
- Ensures slot index is restored even before/without selfplay manager replay.
- Removes a subtle partial-restore gap for nonstandard resume paths.

Re-validation run (small smoke config):
- Built extension: `python setup.py build_chess --inplace`
- Trained short run to checkpoint in `/tmp/puffer_resume_smoke`
- Resumed from explicit model path:
  - `--load-model-path /tmp/puffer_resume_smoke/puffer_chess/1771917694185/model_puffer_chess_000004.pt`
  - Loader auto-selected sibling full state and printed:
    - `Resumed full trainer state from .../trainer_state_full.pt`
- Resumed from `--load-model-path latest` in same data dir:
  - Also resolved to latest `trainer_state_full.pt` and restored successfully.

---

## Session 15 — 2026-02-24: Move Tutor Implementation

### What Was Built

Move tutor reward system: when env resets to a DeepMind curriculum FEN, the expert's
pre-computed Stockfish best move is loaded as a target. Learner gets bonus reward for
matching the expert's piece selection (+0.05) and destination (+0.15). No live Stockfish
needed — zero SPS cost.

### Key Design Decisions

1. **Packed move format**: `uint16_t = from_sq | (to_sq << 6) | (promo << 12)`. Fits in
   2 bytes per position. 2M positions = 4MB memory.

2. **Color forcing**: `learner_color = pos.sideToMove` for tutor episodes. Uses 100% of
   data (no discarding positions where "wrong" side moves). ~50/50 white/black naturally.

3. **Square perspective flip**: Expert moves stored as absolute squares. When learner plays
   Black, both from/to are flipped via `sq ^ 56` to match learner-perspective action space.

4. **Two-phase tracking**: `tutor_phase` tracks piece selection (phase 0) vs destination
   (phase 1) vs done (phase 2). Only the first move attempt in each tutor episode is scored.

5. **Fallback**: If `fens_moves_deepmind.txt` doesn't exist, falls back to FEN-only file.
   Tutor rewards become 0 but curriculum still works.

### Files Changed

- `tools/extract_deepmind_fens_with_moves.py` — NEW: parallel .bag extractor
- `pufferlib/ocean/chess/binding.h` — parse_uci_to_packed, load with moves, config, logging
- `pufferlib/ocean/chess/chess.h` — struct fields, c_reset, process_player_action, c_step
- `pufferlib/config/ocean/chess.ini` — fen_curric_pct=0.5, deepmind_fen_pct=1.0, tutor rewards

### Build: PASS (`python setup.py build_chess`)
