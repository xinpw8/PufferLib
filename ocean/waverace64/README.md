# Wave Race 64 training environment and state evaluator

This environment integrates a statically recompiled Wave Race 64 US Rev 1 core with the native PufferLib 5.0 trainer. The game simulation executes as native host CPU code. PufferLib runs policy inference, rollout tensors, and learning on CUDA. Every environment instance owns its guest memory and suspended game context.

The supported task is deliberately narrow: rider 0, one rider, Sunny Beach, Time Trials, three laps, and one deterministic race-start snapshot. Original N64 rendering, audio, RSP/RDP work, controller-pak behavior, multiplayer, other riders, other courses, and other modes are outside the environment contract. Human evaluation uses a custom state renderer described below. It does not reproduce the cartridge framebuffer.

## Status against the original assignment

The original assignment required a native Wave Race 64 core in the current PufferLib 5.0 stack, high training throughput, demonstrated parity with the game, and preferably a CUDA simulator that avoids CPU simulation work.

| Requirement | Current status |
| --- | --- |
| Native Wave Race 64 core | Implemented with 1,203 statically recompiled game functions and a headless libultra runtime. |
| Current PufferLib 5.0 integration | Implemented against upstream `5.0` commit `ba238f8c` using the native C++17/CUDA trainer. |
| Parity with the game | Proven for selected authoritative state on one pinned deterministic interpreter trace through a native failure terminal. Broader parity remains unproven. |
| High-throughput training | **MEASURED FOR TWO FULL CURRENT-CONTRACT RUNS.** Seeds 707 and 708 completed 5,242,880 decisions at 30,777.828655 and 30,713.28 decisions/s by trainer uptime. Simulation remains CPU-bound. |
| Learning quality | **STOCHASTIC LEARNING DEMONSTRATED IN BOTH FULL-RUN SEEDS.** Seeds 707 and 708 finished 386/514 (`0.750973`) and 410/515 (`0.796117`) stochastic episodes. Deterministic per-head argmax is sharply seed-sensitive: 128/128 for seed 707 and 0/128 for seed 708. A broader seed sweep remains pending. |
| Human-readable evaluation | Implemented as an eval-only state renderer using authoritative rider, course, outcome, and water state. It is a purpose-built visualization, not original Wave Race graphics. |
| CUDA, CPU-free simulation | Unimplemented. The policy and learner use CUDA; `libwr64.a` executes on host CPU workers. |

## Execution architecture

```text
CUDA policy and learner
        | actions and rollout tensors
        v
PufferLib CPU vector workers
        | five controller heads, 55 observations, reward, terminal
        v
waverace64.h adapter
        | WRPad and public runtime ABI
        v
libwr64.a on the host CPU
        | native recompiled game code and headless OS shims
        v
per-instance 8 MiB RDRAM, native stack, ucontext, and snapshot
```

Training is unpaced. The guest's simulated-time cadence does not limit wall-clock throughput. GPU utilization cannot remove the CPU simulator cost in this architecture. Two complete 55-observation training runs are documented below; a broader seed distribution and a fully duration-matched comparison with the retained 43-input baseline remain unavailable.

The adapter follows the native Puffer environment interface in [`waverace64.h`](waverace64.h): `puf_init`, `puf_reset`, `puf_step`, `puf_close`, `puf_log`, and `puf_render`. Training is renderer-free: the training path never calls `puf_render`, never allocates the renderer `Client`, never captures the 33 by 33 display mesh, and never submits Raylib draw calls. It still runs the recompiled simulator and computes the 12 water observation samples on host CPU. Interactive evaluation calls `puf_render` and therefore has evaluation-only CPU and graphics work. The ASCII utility in the runtime repository remains a telemetry plot and is not the state evaluator.

## Fixed scenario and time base

Initialization drives the real menu state machine to this reset contract:

| Field | Required value |
| --- | --- |
| ROM | Wave Race 64 US Rev 1 |
| Game state | `0x28`, racing |
| Race ready | `1` |
| Mode state | `3` |
| Mode | Time Trials, `0` |
| Course | Sunny Beach, `1` |
| Players and riders | `1`, `1` |
| Active rider | Rider `0` |
| Target laps | `3` |

The independent interpreter trace establishes one guest game update every three video-interface callbacks. The video interface is approximately 60 Hz, while controller and gameplay updates occur at 20 Hz on this task. `wr_env_step(machine, pad, 1)` therefore advances one 20 Hz game update.

Production uses `frameskip = 4`. One policy action is held for four guest updates, or 0.2 s of simulated time. This gives five policy decisions per simulated second. `episode_length` and the safety limit count guest updates, not policy decisions. The 14,400-update safety cap corresponds to 720 s. The game's native Time Trials timeout ordinarily fires earlier.

## ROM and runtime requirements

The ROM is user-supplied and is never included in this repository.

| Identity | Required value |
| --- | --- |
| Size | 8,388,608 bytes |
| CRC32 | `394948C4` |
| SHA-1 | `508dfc2d4caa42b6f6de5263d0aed5e44ac7966a` |
| SHA-256 | `f35d2423ebcb86eaf86fa935b613c7532b123a7bc50fb74996984c3b02fc3999` |

The runtime validates the exact US Rev 1 image before boot. A wrong revision, byteswap format, truncated image, or modified image is rejected.

The environment requires 64-bit Linux, a working OpenMP toolchain, the PufferLib 5.0 CUDA dependencies, and a built WR64 runtime tree containing:

- `libwr64.a`
- `runtime/wr_runtime.h`
- `runtime/wr_env.h`
- `RecompiledFuncs/recomp.h`

The public runtime ABI is version `0x00010000`. `WR_RUNTIME_ABI_CHECK()` compares the version and the compiled sizes of `WRMachine`, `WRSnapshot`, and `WRPad` before the archive can write into caller-owned objects. Rebuild the archive and Puffer binary together after any public runtime structure or contract change.

Each `WRSnapshot` owns saved host pointers into one machine's native stack and `ucontext_t`. An initialized `Env` must never move in memory. The custom vector initializer allocates the final environment array once, boots instances in place, preserves per-machine stacks and contexts, and shares a sealed reset RDRAM backing through private copy-on-write mappings. It also restores caller CPU affinity after OpenMP initialization so later rollout threads do not inherit a one-core mask.

## Portable build, test, train, and eval

Set explicit paths so the same commands work from any checkout location.

```sh
export PUFFER_DIR=/path/to/pufferlib-5.0
export WR64_DIR=/path/to/wr64-recomp
export WR64_ROM=/path/to/baserom.us.rev1.z64
```

Verify the ROM and build the runtime archive:

```sh
printf '%s  %s\n' \
  f35d2423ebcb86eaf86fa935b613c7532b123a7bc50fb74996984c3b02fc3999 \
  "$WR64_ROM" | sha256sum --check

cd "$WR64_DIR"
./build_lib.sh
./runtime/run_isolation_acceptance.sh "$WR64_ROM" 512 3
```

Build the native trainer first. This also obtains the Raylib headers used by `pufferenv.h`:

```sh
cd "$PUFFER_DIR"
WR64_DIR="$WR64_DIR" ./build.sh waverace64
```

Build and run the deterministic adapter harness:

```sh
cd "$PUFFER_DIR"
case "$(uname -m)" in
  x86_64|amd64) RAYLIB_DIR="$PUFFER_DIR/raylib-5.5_linux_amd64" ;;
  aarch64|arm64) RAYLIB_DIR="$PUFFER_DIR/raylib-5.5_linux_aarch64" ;;
  *) echo "unsupported architecture" >&2; exit 1 ;;
esac
test -f "$RAYLIB_DIR/include/raylib.h"

g++ -O3 -std=gnu++17 -march=native -flto=auto \
  -ffp-contract=off -fopenmp -DPLATFORM_DESKTOP \
  -I"$PUFFER_DIR" \
  -I"$PUFFER_DIR/ocean/waverace64" \
  -I"$PUFFER_DIR/src" \
  -I"$RAYLIB_DIR/include" \
  -I"$WR64_DIR/runtime" \
  -I"$WR64_DIR/RecompiledFuncs" \
  "$PUFFER_DIR/tests/test_waverace64.cpp" \
  "$WR64_DIR/libwr64.a" \
  -ldl -lpthread -lm \
  -o /tmp/test_waverace64

env OMP_PLACES=cores OMP_PROC_BIND=close \
  /tmp/test_waverace64 "$WR64_ROM"
```

Train a new policy with the checked-in configuration:

```sh
cd "$PUFFER_DIR"
./puffer train --env.rom_path="$WR64_ROM"
```

The configured run contains exactly 5,242,880 policy decisions, or 640 complete batches at 128 agents and horizon 64. Each environment begins with a one-lap target, advances to two laps after one official finish, and advances to three after the next official finish. Curriculum state is local to each environment. Configuration values alone are not evidence; the completed current-contract run and fresh evaluations are reported below.

The checked-in configuration requests at least 32 post-train evaluation episodes. Post-train evaluation reuses the existing 128-agent training vector; `base.eval_agents = 32` applies when a standalone evaluation vector is created. Before post-train evaluation, the trainer waits for rollout workers, forces every Wave Race instance to the official three-lap target, resets every environment, clears transition state, uploads the reset observations, zeros recurrent state, reinitializes action RNG state, and synchronizes the CUDA stream. Fresh CUDA evaluation, post-train evaluation, and the standalone CPU evaluator all use this same three-lap boundary. Evaluation cannot inherit one-lap curriculum state.

A fresh-process evaluation of a named checkpoint remains the acceptance method because it isolates the process lifecycle and pins the artifact under test. The checkpoint must have been trained with `OBS_SIZE = 55`. A 43-input checkpoint has a different first-layer parameter shape and is incompatible; the CUDA and CPU loaders do not define a padding or migration rule. Two compatible checkpoints are retained below. This command selects seed 707, the stable deterministic human-evaluation policy. Do not use `latest`, which selects by filesystem creation time and can pick an unrelated or incompatible checkpoint:

```sh
export WR64_CHECKPOINT=/home/spark-advantage/wr64-results/state-eval-obs55-e0a8e2d4/checkpoints/waverace64/state-eval-obs55-s707/0000000005242880.bin
./puffer-wr64-final eval "$WR64_CHECKPOINT" --headless \
  --base.eval_episodes=100 \
  --base.eval_agents=128 \
  --env.rom_path="$WR64_ROM"
```

`puffer-wr64-final` is the retained deployment binary name. A default local `./build.sh waverace64` build is named `puffer` unless an output name is supplied.

Command-line `--section.key=value` values override [`config/waverace64.ini`](../../config/waverace64.ini). The trainer injects `train.gamma` into the environment discount, keeping the failure time cost and the optional potential-shaping modes consistent with the learner.

The adapter opts out of PufferLib's default `[-1, 1]` learner-side reward clamp. Clipping would change the configured miss, finish, and failure magnitudes. The learner therefore receives emitted rewards unchanged.

Evaluation checks the episode target after complete rollout horizons, so `base.eval_episodes` is a minimum and parallel evaluation can overshoot it. Report the actual `CUDA_EVAL games=N` denominator. For Wave Race, the machine-readable CUDA and CPU lines include deterministic or stochastic mode, exact success count, checkpoints, misses, every terminal-cause rate, target laps, and three-lap success. A retained result must use the native CUDA-policy binary, a fresh process, and an explicit 55-input checkpoint. `base.eval_deterministic=1` selects per-head argmax; the default samples from all five policy heads.

## Human state evaluator and capture

The interactive evaluator is a state-based visualization of the same simulation used for training. Its immutable, pointer-free capture contains the rider position, finite-difference velocity, heading and full body basis, controller state, lap and route state, checkpoint and miss counts, recovery and terminal flags, the authoritative course graph and signed pass points, and a rider-centered 33 by 33 water-height grid at 128 game-unit spacing. The renderer displays that state as a low-poly water surface, course line, colored buoys and pass-side markers, a schematic rider and watercraft, a chase camera, a minimap, and a race HUD.

This makes the control state legible to a human, but it is not graphical parity with the N64 game. The camera, water shading, wake history, rider and watercraft geometry, HUD, and minimap are renderer-owned presentation. No cartridge textures, models, framebuffer, audio, RSP output, or RDP output are used. Renderer correctness means that the shown geometry and labels are projections of captured authoritative fields, within the parity boundary documented below. It does not mean pixel equivalence to Wave Race 64.

Run the CUDA-policy evaluator in a visible window:

```sh
./puffer-wr64-final eval \
  /home/spark-advantage/wr64-results/waverace64-obs55-human-eval.bin \
  --base.eval_deterministic=1 \
  --env.rom_path=/home/spark-advantage/baserom.us.rev1.z64
```

The deployed human-eval artifact remains seed 707 because its deterministic argmax policy completed 128/128 fresh CUDA episodes. Seed 708 is a strong stochastic policy, but its deterministic per-head argmax trajectory completed 0/128 and is not the stable interactive default.

The evaluator uses one environment and advances at five policy decisions per wall-clock second. Rendering targets 60 frames/s. Hold Left Shift to replace policy actions with keyboard input; releasing it returns control to the policy.

| Input while holding Left Shift | Controller action |
| --- | --- |
| W | A, throttle |
| A or Left Arrow | Full left stick X |
| D or Right Arrow | Full right stick X |
| Up Arrow | Full forward stick Y |
| Down Arrow | Full back stick Y |
| S | B, wave damping |
| Space | R, water-surface slide |
| Close window | End evaluation |

For a CPU-only policy viewer or PNG capture, build the standalone evaluator under a distinct output name:

```sh
cd "$PUFFER_DIR"
WR64_DIR="$WR64_DIR" ./build.sh waverace64 wr64_eval --cpu

export WR64_CHECKPOINT=/home/spark-advantage/wr64-results/waverace64-obs55-human-eval.bin
WR64_RENDER_WIDTH=1280 WR64_RENDER_HEIGHT=720 \
  ./wr64_eval "$WR64_CHECKPOINT" \
  --base.eval_deterministic=1 \
  --env.rom_path="$WR64_ROM"
```

The default window is 960 by 540 pixels. `WR64_RENDER_WIDTH` values below 640 and `WR64_RENDER_HEIGHT` values below 480 are ignored.

The CPU evaluator has Wave Race compiled in. Its first positional argument is the checkpoint path; no `waverace64` environment-name token is required.

Capture a deterministic sequence after creating the destination directory:

```sh
export WR64_CAPTURE_DIR=/path/to/wr64-capture
mkdir -p "$WR64_CAPTURE_DIR"

WR64_RENDER_WIDTH=960 WR64_RENDER_HEIGHT=540 \
  ./wr64_eval "$WR64_CHECKPOINT" \
  --env.rom_path="$WR64_ROM" \
  --base.eval_deterministic=1 \
  --capture-dir="$WR64_CAPTURE_DIR" \
  --capture-count=150 \
  --capture-every=12
```

This writes `frame-000000.png` through `frame-000149.png`. At 60 rendered frames/s and five policy decisions/s, `--capture-every=12` retains one image per policy decision and spans 30 simulated seconds. The retained learned-policy capture contains 150 distinct consecutive 960 by 540 images; frame 149 shows target gate 51, 13 gates cleared, and zero misses. `--capture-every=N` keeps every Nth rendered frame. `--capture-hidden` hides the window but still requires a working display and OpenGL context. `--capture-fast` removes normal render pacing and must not be used as a training-throughput measurement. Capture cannot be combined with `--headless`; `--headless` selects metrics-only CPU evaluation and does not call the renderer.

## Action space

The action space is `MultiDiscrete({15, 9, 2, 2, 2})`. Every selected controller state is held through the full internal frameskip.

| Head | Size | Controller input |
| ---: | ---: | --- |
| 0 | 15 | Stick X detents: `-80,-68,-56,-44,-32,-20,-10,0,10,20,32,44,56,68,80` |
| 1 | 9 | Stick Y detents: `-80,-56,-32,-12,0,12,32,56,80` |
| 2 | 2 | A button, throttle |
| 3 | 2 | B button, wave damping |
| 4 | 2 | R button, water-surface slide |

The [official Nintendo controller page](https://www.nintendo.com/eu/media/downloads/games_8/emanuals/nintendo_8/Manual_Nintendo64_WaveRace64_EN.pdf#page=5) identifies A as throttle, B as wave damping, R as sliding, and the stick as handling and center-of-gravity control. It also states that Z duplicates A. The learner therefore has one throttle head and the adapter fixes Z off. Start and camera controls are unnecessary after reset into an active race.

## Observation space

The observation is a flat vector of 55 `float` values. Position, rider, course, miss, finish, recovery, and water fields come from decompilation-derived structures and game memory. Velocity is finite-difference motion per guest update over the selected frameskip. Teleports, recovery transitions, invalid identity, and non-finite motion are excluded from the motion estimate. Any remaining non-finite feature is replaced with zero.

`WR64_SPEED_SCALE` is `55.555557` game units per guest update. `route_total` is the authoritative Sunny Beach route length accumulated from course-node lengths. Basis-vector names remain ordinal because the decompilation does not assign stable body-axis semantics to the three triplets.

| Index | Field | Definition |
| ---: | --- | --- |
| 0 | world X | `x / 1000` |
| 1 | world Z | `z / 1000` |
| 2 | velocity X | Per-update `vx / WR64_SPEED_SCALE` |
| 3 | velocity Z | Per-update `vz / WR64_SPEED_SCALE` |
| 4 | heading X | Normalized forward X from the active physics object |
| 5 | heading Z | Normalized forward Z from the active physics object |
| 6 | relative height | `(y - race_start_y) / 100` |
| 7 | signed slip | `(heading_x * vz - heading_z * vx) / horizontal_speed`, or zero at rest |
| 8 | horizontal speed | `hypot(vx, vz) / WR64_SPEED_SCALE` |
| 9 | applied A | `0` or `1` |
| 10 | applied B | `0` or `1` |
| 11 | reserved Z | Always `0` in the five-head action contract |
| 12 | applied R | `0` or `1` |
| 13 | applied stick X | Signed stick value divided by `80` |
| 14 | applied stick Y | Signed stick value divided by `80` |
| 15 | elapsed fraction | Guest update count divided by `14,400` |
| 16 | current buoy side | `-1` for node type 0, `+1` for type 1, otherwise `0` |
| 17 | node-center forward | Unit direction to the current route node center, projected onto heading |
| 18 | node-center lateral | Unit direction to the current route node center, signed in rider-local space |
| 19 | node-center distance | Distance to current node center divided by `route_total`, clipped to `[0,1]` |
| 20 | within-lap route fraction | Centerline projection on the authoritative route, divided by `route_total` |
| 21 | live misses | Official miss count multiplied by `0.2` |
| 22 | lap fraction | Official lap count divided by target laps, clipped to `[0,1]` |
| 23 | accumulated progress fraction | Signed episode progress divided by `route_total * target_laps`; not clipped |
| 24 | pass-point forward | Unit direction to the signed buoy pass point, projected onto heading |
| 25 | pass-point lateral | Unit direction to the signed buoy pass point in rider-local space |
| 26 | pass-point distance | Distance to the signed pass point divided by `route_total`, clipped to `[0,1]` |
| 27 | next-pass forward | Unit direction to the following signed pass point, projected onto heading |
| 28 | next-pass lateral | Unit direction to the following signed pass point in rider-local space |
| 29 | next-pass distance | Distance to the following pass point divided by `route_total`, clipped to `[0,1]` |
| 30 | node-type-4 flag | `1` only when the current authoritative node type is `4` |
| 31 | recovery class | `0` normally; `0.5` for physics state 24; `1` for a nonzero recovery halfword, state 23, or state 7 before frame 56 |
| 32 | checkpoint fraction | Successful checkpoint events divided by `route_nodes * target_laps` |
| 33 | velocity Y | Per-update `vy / WR64_SPEED_SCALE` |
| 34 | basis 0 X | Active physics body-basis triplet 0, X component |
| 35 | basis 0 Y | Active physics body-basis triplet 0, Y component |
| 36 | basis 0 Z | Active physics body-basis triplet 0, Z component |
| 37 | basis 1 X | Active physics body-basis triplet 1, X component |
| 38 | basis 1 Y | Active physics body-basis triplet 1, Y component |
| 39 | basis 1 Z | Active physics body-basis triplet 1, Z component |
| 40 | basis 2 X | Active physics body-basis triplet 2, X component |
| 41 | basis 2 Y | Active physics body-basis triplet 2, Y component |
| 42 | basis 2 Z | Active physics body-basis triplet 2, Z component |
| 43 | water, forward -64, lateral -96 | Rider-relative water height at that offset |
| 44 | water, forward -64, lateral 0 | Rider-relative water height at that offset |
| 45 | water, forward -64, lateral +96 | Rider-relative water height at that offset |
| 46 | water, forward +64, lateral -96 | Rider-relative water height at that offset |
| 47 | water, forward +64, lateral 0 | Rider-relative water height at that offset |
| 48 | water, forward +64, lateral +96 | Rider-relative water height at that offset |
| 49 | water, forward +192, lateral -96 | Rider-relative water height at that offset |
| 50 | water, forward +192, lateral 0 | Rider-relative water height at that offset |
| 51 | water, forward +192, lateral +96 | Rider-relative water height at that offset |
| 52 | water, forward +384, lateral -96 | Rider-relative water height at that offset |
| 53 | water, forward +384, lateral 0 | Rider-relative water height at that offset |
| 54 | water, forward +384, lateral +96 | Rider-relative water height at that offset |

For buoy node types 0 and 1, the pass point is the node center plus a signed `400`-unit offset along the course node's decomp-derived lateral vector. Earlier adapters exposed only the node center, which withheld the required passing side from the policy.

For indices 43 through 54, let rider position be `(x, y, z)`, normalized horizontal heading be `(hx, hz)`, forward offset be `f`, and lateral offset be `l`. The sampled point and feature are:

```text
sample_x = x + f*hx + l*hz
sample_z = z + f*hz - l*hx
observation = (water_height(sample_x, sample_z) - y) * 0.01
```

The row order is forward offset `-64, 64, 192, 384`, with lateral offset `-96, 0, 96` inside each row. Units before scaling are game units. These 12 values expose nearby dynamic wave height without rendering an image or supplying the dense 33 by 33 evaluator mesh to the learner.

Changing from 43 to 55 observations changes the policy ABI. Every prior 43-input checkpoint has an incompatible first layer, even though observations 0 through 42 retain their meanings. There is no supported conversion or zero-padding path. Use the retained 55-input checkpoint documented below or train another model with 55 inputs.

These 55 fields are a compact control observation, not the full game state. The environment does not claim strict Markov sufficiency across every Wave Race mechanic.

## Reward

The checked-in coefficients are:

| Term | Coefficient |
| --- | ---: |
| Speed | `0` |
| New route frontier | `3` per lap of newly reached route distance |
| Slip | `0` |
| Successful checkpoint | `0.3` |
| Miss | `-0.5` per official miss event |
| Nonterminal time cost | `-reward_fail * (1 - gamma)` per policy transition |
| Official finish | `+10` |
| Failure | `-2` |

The checked-in configuration uses reward mode 2. It credits only a new maximum route frontier and verified checkpoint events:

```text
frontier_gain = max(0, max_progress_after - max_progress_before)
shaping = 3 * frontier_gain / route_total
        + 0.3 * successful_checkpoint_events
```

The maximum frontier is monotone, so reversing and revisiting a segment cannot earn it twice. A route-node advance accompanied by an official miss earns no checkpoint term. Recovery, teleport, invalid-identity, and discontinuous route transitions earn neither motion nor frontier credit. This matches the practical Puffer racing pattern of dense distance progress plus discrete gates while adding a one-credit frontier guard against oscillation.

Reward mode 0 retains strict terminal-cancelled potential shaping as an ablation. Mode 1 retains terminal potential. Both use `train.gamma` as the environment discount. Mode 2 is the current configured objective. Earlier mode comparisons used the superseded 43-observation ABI and are not learning evidence for the current policy.

The nonterminal time cost removes a discount loophole found in pilot training. If `F` is the configured failure magnitude, the discounted sum of `-F * (1 - gamma)` on every nonterminal transition plus `-F` at a failure terminal is exactly `-F`, regardless of episode duration. Stalling until the native timeout therefore cannot make failure cheaper. A successful trajectory retains discounted base return `-F + (F + finish) * gamma^(T-1)`, so faster official finishes remain preferable.

Optional instantaneous speed and slip terms exist for experiments. The checked-in configuration keeps both coefficients at zero. Motion terms are suppressed across game-driven teleports and recovery transitions.

### Lap curriculum and evaluation boundary

Training starts each vector instance at one lap and advances that instance from 1 to 2 to 3 laps after one official success at each level. The curriculum changes only the two verified native lap-count words after restoring the same race-start snapshot. Actions, observations, physics, course, rider, and terminal definitions are unchanged. Affine Lock, Bat, and Clifford provide native Puffer precedents for success-driven per-environment curricula.

Every evaluation forces three laps before reset. `target_laps=3` and equality between `three_lap_success_rate` and `success_rate` are acceptance invariants. The adapter regression checks this reset boundary, and the retained current-contract evaluations report `target_laps=3`. Future accepted results must continue to report both fields.

## Terminal and autoreset contract

| Outcome | Condition | Reward/log classification |
| --- | --- | --- |
| Success | Official `RiderStruct.finished` flag and no disqualification | Finish reward; `success_rate` |
| Native failure | Disqualification, or generic official end without finish | Failure penalty; disqualification is logged separately from generic failure |
| Safety timeout | 14,400 guest updates without an official terminal | Failure penalty; `safety_timeout_rate` |
| Environment fault | Race identity or game state leaves the fixed contract without an official terminal | Fatal diagnostic and process abort |

On an ordinary terminal, `puf_step` records the episode log, sets terminal to `1`, restores the exact race-start snapshot, and returns observations from the new episode. The terminal reward and terminal flag remain in their transition buffers through autoreset. PufferLib initializes those buffers separately before the first step.

## Log fields

PufferLib sums per-environment `Log` structures, divides by the internal episode count `n`, and publishes the following `env/*` means:

| Field | Meaning |
| --- | --- |
| `perf` | Maximum forward progress divided by `route_total * target_laps`, clipped to `[0,1]` |
| `score` | Signed accumulated route progress in game units |
| `distance` | Stable-motion 3D path length; teleports and recovery discontinuities are excluded |
| `checkpoints` | Successful route-node advances after subtracting simultaneous misses |
| `misses` | Accumulated official miss events |
| `success_rate` | Fraction of logged episodes with an official finish |
| `failure_rate` | Fraction with a generic official failure, excluding disqualification and safety timeout |
| `disqualification_rate` | Fraction ending in official disqualification |
| `safety_timeout_rate` | Fraction ending at the adapter's 14,400-update safety cap |
| `env_fault_rate` | Diagnostic field for contract failures; the current fatal path aborts before a normal episode log |
| `mean_speed` | `20 * distance / episode_length`, in game units per second |
| `episode_return` | Undiscounted sum of emitted rewards |
| `episode_length` | Guest updates, not policy decisions |
| `target_laps` | Episode target, averaged over completed episodes; training can mix curriculum levels and evaluation must equal `3` |
| `three_lap_success_rate` | Fraction of episodes that both targeted three laps and reached an official finish |
| `n` | Episode count used for aggregation; added by the PufferLib logger |

## Correctness evidence

The independent oracle uses Mupen64Plus pure interpreter commit `6dca4c15370ac3e2171ce7b31426695f8f39b460` and RSP HLE commit `8a7a472a7172eb2c8725b305eae26818ed7b51a2`. It boots the same ROM through real menus, applies the same guarded controller policy independently, and shares no static runtime code with the probe.

After aligning one extra pre-game interpreter controller scan, 638 controller scans cover guest frames 0 through 637. The following are bit-exact at every aligned scan:

- controller buttons and signed stick values;
- game frame, state, ready state, mode state, course, mode, player count, rider count, node count, and active rider;
- the primary game RNG at `0x800D4640`;
- active physics X/Y/Z, forward X/Z, and speed float bits;
- lap, target node, misses, disqualification, ended, finished, and recovery.

Both executions reach the same route events and the native Time Trials failure state `0x29` at guest frame 637, with one miss and matching official end flags. The runtime's first `osGetTime` seed was corrected to the interpreter-measured low word `0x56D84BE4`, which made the primary RNG and the authoritative fields above exact.

The evidence has explicit limits. It covers one deterministic Sunny Beach Time Trials trace through failure. It does not cover every controller trace, successful three-lap completion, other courses, other modes, graphics, audio, controller-pak behavior, or every writable RDRAM byte. A secondary `Math_Rand` state at `0x80226F00` differs from guest frame 2 because post-reset `osGetTime` advancement is not modeled. That stream did not alter the authoritative fields above through frame 637.

The runtime's retained isolation test initializes 16 machines, compares serial and parallel controller streams, repeats the parallel run three times, and requires exact final RDRAM, stack, recomp context, machine state, and trajectory hashes. The log is retained only on the deployment host at `/home/spark-advantage/wr64-recomp/runtime/isolation_acceptance_20260822T050325Z_3003381.log`; its SHA-256 is `805b8384d97306f77c043734727a5cd0ffbece35088574bbb4ebf9ba165d852f`.

The adapter regression harness in [`tests/test_waverace64.cpp`](../../tests/test_waverace64.cpp) covers the 55-value ABI shape, exact placement of the 12 water features, action mapping, internal frameskip, body basis, reset contract, guest halfword lane, buoy side, lap-wrap continuity, official misses and disqualification, discount-correct failure shaping, shortened one-lap official finish, full three-lap scripted reachability, B/stick-Y interventions, observation ranges, deterministic baselines, and vector affinity/ownership. The unit harness does not invoke the central CUDA train/evaluation path.

The checked-in exact-water test compares `wr64_water_height` bit-for-bit against the recompiled cartridge function `func_8004D30C` at 4,096 deterministic points on the live water field. It exercises both interpolation branches, invokes the reference on scratch RDRAM because the recompiled function uses a guest stack, and verifies that the pure sampler does not change live RDRAM. The implementation was also characterized at 262,144 points across 64 randomized complete water fields with zero float-bit mismatches when compiled with `-ffp-contract=off`.

That result is exact equivalence to the recompiled `func_8004D30C`, not an independent emulator comparison of wave evolution. It validates the query used by observations 43 through 54 and the evaluator mesh. It does not extend the held-A interpreter parity trace to other wave states or inputs.

The render-state regression captures the state twice, requires byte-identical logical content and hashes, checks every course pass point, checks every value in the 33 by 33 water tile bit-for-bit against `wr64_water_height`, verifies live RDRAM is unchanged, and verifies that capture alone leaves `env->client` null. It validates the state projection, not Raylib raster output or N64 visual parity.

The separate renderer regression in [`tests/test_waverace64_render.cpp`](../../tests/test_waverace64_render.cpp) passed twice independently with identical results on the tested software-rendered stack:

| Check | Retained result |
| --- | --- |
| Training path stays lazy | `Client` remained null through 192 decisions, including autoresets |
| Pure authoritative capture | State hash `7d5f6487fbfc38ba`, 59 course nodes |
| Renderer preserves simulator core | Eight repeated `puf_render` calls left RDRAM, stack, recomp context, machine scalars, adapter state, and TLS owner unchanged |
| Fixed-state pixels | Two 960 by 540 captures were byte-identical; pixel hash `1eef85fa7d51633f` |
| Render cadence independence | Headless and variably rendered 96-decision trajectories matched; hash `ffa76503834a3bff` |
| Moving capture | 48 rendered decisions reached tick 48; final state hash `9d18db27cdfc907d` |

The pixel hash is a regression baseline for the tested Raylib/OpenGL software stack. It is not asserted to be portable across graphics drivers, and it is not a reference N64 framebuffer hash.

### Adapter acceptance and remaining parity gaps

| Gate | Status |
| --- | --- |
| Shortened one-lap official finish fixture | **PASS:** official finish at update 1,070 with action hash `c6ae00920fd86802`. |
| Unmodified three-lap scripted reachability through the public observation interface | **REGRESSION COVERAGE:** the checked-in harness exercises the fixture with the 55-value observation buffer; this is controller reachability, not policy learning. |
| B and stick-Y authoritative interventions | **PASS:** B changed the authoritative trace in 5 of 6 probes; stick Y changed it in all 6. |
| Generic Puffer regression | **PASS:** stock CartPole CPU evaluation repeated byte-for-byte at fixed seed; stock Breakout CUDA build and evaluation exited zero. |
| Multi-trace interpreter parity, including B and R action regimes | **PENDING** |
| Post-reset secondary RNG and whole-RDRAM parity | **PENDING** |

The deterministic controller fixtures establish task reachability and regression behavior. They do not establish policy learning or current training performance.

## Performance and learning acceptance

The current contract has 55 observations. Two complete renderer-free runs were measured:

| Training seed | Decisions | Trainer uptime | Decisions/s | Guest updates/s | Process wall | Wall decisions/s | CPU | Maximum RSS | Internal GPU mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 707 | 5,242,880 | 170.346 s | 30,777.828655 | 123,111.314618 | 171.28 s | 30,609.995329 | 1,687% | 1,625,884 KiB | Not retained |
| 708 | 5,242,880 | 170.704 s | 30,713.28 | 122,853.13 | 171.65 s | 30,544.01 | 1,687% | 1,625,936 KiB | 4.287% |

Seed 707 ran `./puffer-wr64-obs55`, SHA-256 `893c63d0cde3fb9d8073232fe5b1c98a543f53e00732c01b5488e7c7817952f`. Seed 708 ran `./puffer-wr64-final`, SHA-256 `e96263824d6083e08f8d18d57aa415dcf68c2ca2a9260d1b8192cbf90a59633c`. Later changes affected renderer-state hashing, the render harness, and documentation rather than the training path, but the two run binaries remain distinct artifacts.

The seed-708 GPU value is the time-weighted internal metric for that run. No equivalent full-run value is retained for seed 707. The renderer and capture paths were inactive in both runs. Seed 707's final logged curriculum batch reported success 1.0 and `target_laps=3`; this is a final-batch training result, not an evaluation statistic.

The retained 43-input historical baseline had mean 30,900.2 and median 30,914.5 policy decisions/s. It is not documented as a fully pinned duration-matched comparison with these runs, so no regression percentage is assigned. The current network has 109,312 parameters versus 107,776 previously, an increase of 1,536 or 1.425%. The evidence does not isolate water-query cost from network, host, placement, or run-protocol effects.

Three earlier 55-input short screens averaged 29,609.44 decisions/s, 16.220 CPU equivalents, 1.55052 GiB mean peak RSS, and 4.384% device-wide GPU utilization. They are retained as preliminary screens, not substituted for the complete runs. The GPU figure is device-wide and was not sampled continuously or attributed to this process. Their provenance is deployed Puffer commit `8a9c51f2cebac95c36119d2eb975d53db10b4529`, local renderer-harness commit `1fa4769b942a6179828a2e7b5d5773e8d3596234`, runtime commit `e3f56302898a98ec7f7b20ca35fc1b5de69fe890`, trainer SHA-256 `893c63d0cde3fb9d8073232fe5b1c98a543f53e00732c01b5488e7c7817952f`, and CPU evaluator SHA-256 `a07089133be8f5c32ef89119313f6702170a98d2566451c82f0b56467a0d1161`.

Both compatible checkpoints are 437,248 bytes:

| Training seed | Checkpoint path | SHA-256 |
| ---: | --- | --- |
| 707 | `/home/spark-advantage/wr64-results/state-eval-obs55-e0a8e2d4/checkpoints/waverace64/state-eval-obs55-s707/0000000005242880.bin` | `a6696e3888ca472712071aa9fd6b82b377e3ddf956db41ca2082488c3145fc59` |
| 708 | `/home/spark-advantage/wr64-results/state-eval-obs55-seed708-final/checkpoints/waverace64/state-eval-obs55-s708/0000000005242880.bin` | `83382a3f31141a1645a5be2cb8c31696480066838faaee70c8527138722027dd` |

Fresh-process evaluation produced:

| Training seed | Evaluator | Action mode | Episodes | Successes | Success rate | Perf | Score | Checkpoints | Misses |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 707 | CUDA | Per-head argmax | 128 | 128 | 1.000000 | 0.937920 | 81,820.898438 | 46.000000 | 0.000000 |
| 707 | CUDA, action seed 7001 | Stochastic | 514 | 386 | 0.750973 | 0.877817 | 76,568.945312 | 39.085602 | 2.708171 |
| 707 | CPU | Per-head argmax | 1 | 1 | 1.000000 | 0.889577 | 77,603.546875 | 39.000000 | 1.000000 |
| 708 | CUDA | Per-head argmax | 128 | 0 | 0.000000 | 0.720770 | 62,625.097656 | 32.000000 | 1.000000 |
| 708 | CUDA, action seed 7002 | Stochastic | 515 | 410 | 0.796117 | 0.938844 | 81,918.937500 | 41.429127 | 3.330097 |

For seed 707, deterministic CUDA reported zero generic failures, disqualifications, timeouts, or faults. Stochastic CUDA reported 80 generic failures (`0.155642`), 48 disqualifications (`0.093385`), and no timeout or fault. For seed 708, all 128 deterministic replicas ended in generic failure, with no disqualification, timeout, or fault. Its stochastic evaluation reported 10 generic failures (`0.019417`), 95 disqualifications (`0.184466`), and no timeout or fault. Every CUDA evaluation reported `target_laps=3`; seed 707 deterministic `three_lap_success_rate` was 1.

Seed 708 is not a non-learning failure. Its stochastic policy completed 410/515 official three-lap episodes, while seed 707 completed 386/514 under a different held-out action seed. The collapse is specific to deterministic per-head argmax: the selected modal trajectory finished 0/128 while seed 707's finished 128/128. Seed sensitivity is therefore demonstrated for argmax extraction, while a broader training-seed distribution remains pending.

The seed-707 deterministic CPU episode succeeded after 619 policy decisions with zero generic failures, disqualifications, timeouts, or faults and `target_laps=3`. It is a functional policy and environment check, not enough evidence for a CPU success-rate estimate or CUDA/CPU action-by-action equivalence. The deterministic CUDA entries are replicas of one deterministic initial condition per checkpoint. The stochastic results characterize two trained seeds under one held-out action seed each.

| Current 55-input acceptance item | Status |
| --- | --- |
| Full renderer-free training throughput | **MEASURED FOR SEEDS 707 AND 708:** 30,777.828655 and 30,713.28 decisions/s by trainer uptime |
| Full-run CPU and memory utilization | **MEASURED FOR BOTH:** 1,687% CPU; maximum RSS 1,625,884 and 1,625,936 KiB |
| GPU utilization | **PARTIAL:** seed 708 time-weighted internal mean 4.287%; comparable seed-707 full-run telemetry not retained |
| Compatible checkpoints | **RETAINED:** both 437,248 bytes with hashes above |
| Fresh stochastic CUDA three-lap evaluation | **PASS IN BOTH SEEDS:** 386/514 (`0.750973`) and 410/515 (`0.796117`) official successes |
| Deterministic per-head argmax | **SHARPLY SEED-SENSITIVE:** seed 707 finished 128/128; seed 708 finished 0/128 |
| Fresh deterministic CPU evaluation | **PASS AS A SEED-707 ONE-EPISODE CHECK:** 1/1 official success in 619 decisions |
| Learned-policy PNG capture | **PASS FOR STABLE SEED 707:** 150 distinct consecutive 960 by 540 frames spanning 30 simulated seconds |
| Broader training-seed statistics | **PENDING:** two full seeds establish the contrast but not the wider distribution |

## Audit

[`AUDIT.md`](AUDIT.md) traces the prior claims, the evidence behind each verdict, the repairs already made, and the remaining CUDA, performance, and learning gaps.
