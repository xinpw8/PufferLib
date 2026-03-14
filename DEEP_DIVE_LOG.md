# Deep Dive Engineering Log

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

### SPC Changes Still Needed:
- [ ] Fix centipawn_eval.py architecture auto-detection
- [ ] Create eval_policy_vs_policy.py (gold-standard: current beats past snapshots)
- [ ] Add mate-in-1 detection reward to c_step in chess.h
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
