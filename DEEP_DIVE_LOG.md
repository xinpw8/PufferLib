# Deep Dive Engineering Log

## Session 22: Policy-vs-Policy Gold Standard Eval (2026-03-14)

### SPC Change: Create eval_policy_vs_policy.py
**Assumption:** The gold standard for selfplay is "can current policy beat past snapshotted policy?" Without this, we can't objectively measure whether training is producing stronger play.
**Change:** Created `eval_policy_vs_policy.py` with:
- `evaluate_policy()`: Plays latest checkpoint against selected historical snapshots (earliest, 25%/50%/75% through training, latest predecessor). Reports W/D/L, win rate, mean/median CP, ELO estimate per matchup.
- `compute_verdict()`: PASS/FAIL/INCONCLUSIVE based on 4 criteria:
  1. Beats earliest checkpoint with >60% WR
  2. Beats midpoint checkpoint with >55% WR
  3. Mean CP advantage across all matchups is positive
  4. No regression (never loses >60% to any snapshot)
- `run_tournament()`: Full round-robin tournament with iterative ELO computation, checks for monotonic ELO increase
- CLI: `--quick` (10 games, depth 8, 4 opponents), `--tournament`, `--dir`, `--checkpoint`
- Saves JSON results per evaluation for tracking over time
- Uses `centipawn_eval.py` infrastructure (load_model with auto-detection, play_game_cp, evaluate_position_cp)
**File:** eval_policy_vs_policy.py
**Verification:** Imports successfully on Spark. Full test requires training checkpoints.
**Status:** APPLIED

---

## Session 21: Mate-in-1 Detection Rewards (2026-03-13)

### SPC Change: Add mate-in-1 detection rewards to chess selfplay
**Assumption:** The agent lacks targeted incentives to create mating threats or avoid blundering into opponent mate threats during full selfplay games. Dense reward shaping for these tactical patterns should accelerate learning to close out winning positions and avoid fatal blunders.

**Change:** Added three configurable reward signals that fire after the learner's move completes during full games (not puzzles or BC episodes):

1. **reward_allowed_mate** (default -0.1): Penalty when, after the learner's move, the opponent has a forced mate-in-1. Uses `find_mate_in_1()` on the resulting position (opponent to move). Detects blunders that allow immediate checkmate.

2. **reward_mate_threat** (default 0.1): Reward when the learner gives check and the opponent has 2 or fewer legal moves. This is a lightweight proxy for "we created a near-mate position" -- exact mate-in-1 threat detection from the learner's perspective would require checking all opponent responses (expensive).

3. **reward_mate_defense** (default 0.05): Config field added but defense detection deferred -- tracking whether a mate threat existed before the learner's move requires cross-step state. Field is wired through init for future use.

**Files modified:**
- `pufferlib/ocean/chess/chess.h`: Chess struct (3 float fields), c_step() (mate detection block before clip_rewards)
- `pufferlib/ocean/chess/binding.h`: my_init() (3 DictItem reads)
- `pufferlib/config/ocean/chess.ini`: Added reward_mate_threat, reward_mate_defense, reward_allowed_mate
- `pufferlib/config/ocean/chess_pure_selfplay.ini`: Same

**Guard conditions:** Only fires when `move_completed && mover == learner_color && !puzzle_drill_mode && curriculum_episode_type != 0`. Rewards are subject to `clip_rewards()` (capped at 0.9).

**Performance note:** `find_mate_in_1()` generates all legal moves and tries each one, which is O(moves * opponent_moves) per call. This runs once per learner move during full games. In typical middlegame positions (~30 legal moves), this is ~900 do_move/undo_move pairs -- lightweight compared to the existing mate-in-N solvers used for puzzle validation.

**Verification:** Check that reward_allowed_mate fires when the learner blunders (should correlate with losses 1-2 moves later). Check that reward_mate_threat fires in winning endgames.

**Status:** APPLIED, not yet verified via training

---

## Session 20: Network Architecture Deep Dive (2026-03-14)

### SPC Change 1: Fix ChessSeven Spatial Architecture
**Assumption:** 1x1 convolution kernels prevent the network from learning spatial relationships between pieces on different squares. A knight on e4 attacking f6 is invisible to a pointwise conv.
**Change:** Replaced 1x1 kernels with 3x3 full convolutions + proper residual:
- conv1: Conv2d(19, 32, 3x3, pad=1) -> ReLU
- conv2: Conv2d(32, 32, 3x3, pad=1) + skip(Conv2d(19, 32, 1x1)) -> ReLU (residual)
- conv3: Conv2d(32, 16, 3x3, pad=1) -> ReLU -> flatten (16*8*8=1024)
- Effective receptive field: 5x5 (vs 1x1+3x3dw=3x3 before)
**Param count estimate:** ~340K total (vs 499K old ChessSeven, vs 6.4M ChessTwo)
**File:** pufferlib/ocean/torch.py (ChessSeven class)
**Verification question:** Does the new architecture learn piece interactions better? Measure: centipawn improvement rate per step.
**Status:** APPLIED, not yet verified via training

### SPC Change 2: Fix SPS Configuration
**Assumption:** 42K SPS caused by run_pure_selfplay.py overriding to 4096 agents (vs ini's 16384). GPU starved at 36% compute time.
**Change:** Updated run_pure_selfplay.py:
- total_agents: 4096 -> 16384
- num_buffers: 4 -> 8
- num_threads: 32 -> 128
- chess_encoder: 2 -> 1 (ChessSeven, smaller model fits VRAM at 16K agents)
- Added reward shaping (material 0.05, check 0.01, mate 5.0, draw -0.5, syzygy 0.5)
**Expected SPS:** 300K+ (based on LEDGER session 14 with smaller model + 16K agents)
**File:** run_pure_selfplay.py
**Verification question:** Does SPS exceed 300K with 16384 agents? Measure: first 5 epochs SPS.
**Status:** APPLIED, not yet verified

### SPC Change 3: Fix centipawn_eval.py Architecture Auto-Detection (2026-03-13)
**Problem:** centipawn_eval.py had a single hardcoded `ChessEncoderPy` matching the old 1x1 pointwise architecture. Loading checkpoints from other architectures caused size mismatches (`encoder.square_embed.weight` vs `conv1.weight` vs `conv4.weight`).
**Change:** Added architecture auto-detection supporting three model variants:
- `detect_architecture(state_dict)` inspects key names:
  - `square_embed` in keys -> `old_1x1` (original ChessSeven)
  - `conv4` + `scalar_fc` in keys -> `chess_two` (ChessTwo with 4x conv3x3, 256ch)
  - `conv1` + `skip` in keys -> `new_3x3` (revised ChessSeven with 3x3 + residual)
- `ChessPolicyOld` wraps old 1x1 encoder (renamed from `ChessPolicy`/`ChessEncoderPy`)
- `ChessPolicyTwo` + `ChessTwoEncoder` mirrors ChessTwo exactly (conv1-4, scalar_fc1/fc2, 16 spatial channels)
- `ChessPolicyNew` mirrors new ChessSeven's 3x3 architecture (conv1->conv2+skip->conv3->proj->actor)
- `load_model()` loads state_dict once, auto-detects arch, infers hidden_size/cnn_channels for ChessTwo, loads weights
- Backward-compatible aliases preserved: `ChessPolicy = ChessPolicyOld`, `ChessEncoderPy = ChessEncoderOld`
**File:** centipawn_eval.py
**Verification:** Tested on Spark -- all three architectures detected correctly, old_1x1 and chess_two checkpoints load and forward pass without errors.
**Status:** APPLIED, VERIFIED

### SPC Changes Still Needed:
- [x] Fix centipawn_eval.py architecture auto-detection
- [x] Create eval_policy_vs_policy.py (gold-standard: current beats past snapshots) (Session 22)
- [x] Add mate-in-1 detection reward to c_step in chess.h (Session 21)
- [ ] Verify all above post-training

---

## PFRN Change 1: Proper Observation Processing Network
**Assumption:** Flat MLP on raw bytes cannot learn semantic meaning. Multi-byte values split across independent linear inputs. Tile grid flattened destroys spatial info. Categoricals treated as continuous.
**Change:** Complete rewrite of pokemon_firered_native_puffer/torch.py:
- Tile grid [107:188]: unpacked to [B,8,9,9] bit-planes, processed by 2-layer Conv2d(8->32->32, 3x3)
- Categoricals: species Embedding(413,16), map Embedding(256,16), moves Embedding(360,8), NPC graphics/movement embedded
- Multi-byte values: properly combined (byte_lo + byte_hi*256), normalized, signed handled
- Binary flags: unpacked from bitmasks to individual features
- LSTM added for episode memory
- Total input: tile_conv(2592) + continuous(133) + embeddings(236) = 2961 -> proj(256) -> fc(256) -> LSTM(256) -> heads
**File:** worktrees/pokemon_firered_native_puffer/pokemon_firered_native_puffer/torch.py
**Verification question:** Does the structured network learn faster than the flat MLP? Measure: reward curve slope in first 100 epochs.
**Status:** APPLIED, not yet verified

### PFRN Changes Still Needed:
- [ ] Step loop optimization analysis (strip GBA emulation overhead)
- [ ] step_frames parameter tuning (1 vs 24)
- [ ] Benchmark SPS with new network
- [ ] Verify observation byte offsets match actual struct layout
