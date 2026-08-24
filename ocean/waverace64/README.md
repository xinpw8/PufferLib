# Wave Race 64 training environment and state evaluator

This environment integrates a statically recompiled Wave Race 64 US Rev 1 core with the native PufferLib 5.0 trainer. The game simulation executes as native host CPU code. PufferLib runs policy inference, rollout tensors, and learning on CUDA. Every environment instance owns its guest memory and suspended game context.

The supported task is deliberately narrow: rider 0, one rider, Sunny Beach, Time Trials, three laps, and a finite pool of authentic cartridge race-start states selected per episode. Original N64 rendering, audio, RSP/RDP work, controller-pak behavior, multiplayer, other riders, other courses, and other modes are outside the environment contract. Human evaluation uses a custom state renderer described below. It does not reproduce the cartridge framebuffer.

## Status against the original assignment

The original assignment required a native Wave Race 64 core in the current PufferLib 5.0 stack, high training throughput, demonstrated parity with the game, and preferably a CUDA simulator that avoids CPU simulation work.

| Requirement | Current status |
| --- | --- |
| Native Wave Race 64 core | Implemented with 1,203 statically recompiled game functions and a headless libultra runtime. |
| Current PufferLib 5.0 integration | Implemented against upstream `5.0` commit `ba238f8c` using the native C++17/CUDA trainer. |
| Parity with the game | Proven for selected authoritative state on one pinned deterministic interpreter trace through a native failure terminal. Broader parity remains unproven. |
| High-throughput training | **PASS.** The selected randomized-wave run completed 10,485,760 decisions at 56,575.81 decisions/s and 113,151.62 guest updates/s. A paired fixed/random trainer benchmark took 43.81/43.02 s. Simulation remains CPU-bound. |
| Learning quality | **PASS FOR THE SUPPORTED TASK.** The randomized-wave seed-903 checkpoint completed 128/128 deterministic episodes on held-out wave seed 2902 and 128/128 on post-selection, training-disjoint wave seed 16902. Stochastic evaluation completed 510/512 on each pool. |
| Human-readable evaluation | Implemented as an eval-only state renderer using authoritative rider, course, native clock, native speed, power, outcome, and water state. Its compact edge HUD follows the actual Time Trials information layout without claiming original Wave Race graphics. |
| CUDA, CPU-free simulation | Unimplemented. The policy and learner use CUDA; `libwr64.a` executes on host CPU workers. |

## Execution architecture

```text
CUDA policy and learner
        | actions and rollout tensors
        v
PufferLib CPU vector workers
        | five controller heads, 57 observations, reward, terminal
        v
waverace64.h adapter
        | WRPad and public runtime ABI
        v
libwr64.a on the host CPU
        | native recompiled game code and headless OS shims
        v
per-instance 8 MiB RDRAM, native stack, ucontext, and snapshot
```

Training is unpaced. The guest's simulated-time cadence does not limit wall-clock throughput. GPU utilization cannot remove the CPU simulator cost in this architecture. The randomized reset contract has controlled renderer-free rollout and paired-trainer comparisons, two complete training runs, fresh policy evaluation on two non-training wave pools, and a full-race capture. One complete fixed-wave OBS57 run and two OBS55 runs remain as explicitly historical evidence.

The adapter follows the native Puffer environment interface in [`waverace64.h`](waverace64.h): `puf_init`, `puf_reset`, `puf_step`, `puf_close`, `puf_log`, and `puf_render`. Training is renderer-free: the training path never calls `puf_render`, never allocates the renderer `Client`, never captures the 33 by 33 display mesh, and never submits Raylib draw calls. It still runs the recompiled simulator and computes the 12 water observation samples on host CPU. Interactive evaluation calls `puf_render` and therefore has evaluation-only CPU and graphics work. The ASCII utility in the runtime repository remains a telemetry plot and is not the state evaluator.

## Fixed scenario and time base

Initialization drives the real menu state machine to this reset contract:

| Field | Required value |
| --- | --- |
| ROM | Wave Race 64 US Rev 1 |
| Game state | `0x28`, racing |
| Race ready | `1` |
| Mode state | `2`, the first live race update |
| Mode | Time Trials, `0` |
| Course | Sunny Beach, `1` |
| Players and riders | `1`, `1` |
| Active rider | Rider `0` |
| Target laps | `3` |
| Native total and lap clocks | `0 ms`, `0 ms` |

The independent interpreter trace establishes one guest game update every three video-interface callbacks. The video interface is approximately 60 Hz, while controller and gameplay updates occur at 20 Hz on this task. `wr_env_step(machine, pad, 1)` therefore advances one 20 Hz game update.

Production uses `frameskip = 2`. One policy action is held for two guest updates, or 0.1 s of simulated time. This gives ten policy decisions per simulated second while retaining the native 20 Hz game cadence. `episode_length` and the safety limit count guest updates, not policy decisions. The 14,400-update safety cap corresponds to 720 s. The game's native Time Trials timeout ordinarily fires earlier.

### Per-episode wave reset

Production sets `randomize_waves = 1` and `wave_variants = 128`. Each pool entry comes from a complete cartridge boot through the native course initialization and its 57-frame live-start chain. A deterministic transform of `wave_seed` and the variant index supplies the initial `osGetTime` value before boot. This preserves the scripted large waves, water velocities, rider contact state, RNG state, and every other RDRAM dependency produced by the cartridge. Calling only the low-level wave generator was rejected because it did not reproduce that coherent race-start distribution.

Pool construction validates every donor against the canonical route cache and snapshot scalar contract. It stores only the donor's changed 4 KiB RDRAM pages, reconstructs every entry byte-for-byte, rejects exact duplicate complete 384 by 128 water fields, then rebases donor snapshots to the canonical copy-on-write backing. Extraction is streamed so a standalone environment retains only one full donor at a time. `wave_variants` must be a power of two from 1 through 128. The checked-in value of 128 supplies the largest validated pool.

`puf_reset` restores the environment owner's own native stack, recompilation context, and canonical snapshot, applies the selected variant pages, writes the curriculum lap target, recomputes the variant-specific vertical origin, and projects the first observation. Selection is an exact per-environment odd-stride permutation: a single environment visits every pool entry once per `wave_variants` resets, and the first reset of environments 0 through K-1 is stratified across K entries. `puf_eval_reset` rewinds the permutation so evaluation is independent of policy-dependent training reset history. The same `wave_seed`, environment index, and reset number reproduce the same state. `wave_seed` is independent of `base.seed`.

Variant choice, page application, and validation occur only during initialization or reset. `puf_step` evolves the selected cartridge state normally and contains no pool-selection or page-copy work. Training does not invoke the renderer. On AArch64, the runtime's opaque floating-point scope canonicalizes FPCR before the adapter's initialization, reset, projection, step, and render work and restores the caller's complete FPCR token afterward. Hostile-FPCR regressions cover reset, step, terminal autoreset, state refresh, and render capture.

Set `randomize_waves = 0` to retain the byte-identical pinned `WR_BOOT_OS_TIME` snapshot used by the interpreter comparison and deterministic controller regressions. For train and held-out studies, use different explicit `wave_seed` values. The fixed index salt `0xB9DCF3C0` is a deterministic tested transform, not a source of cryptographic randomness. The schedule test found 128 unique boot-time, full-RDRAM, and complete-water hashes for the checked-in seed 42 and candidate training seed 902, with zero overlap between them. It also found the same uniqueness and zero overlap for candidate train/held-out seeds 902 and 2902. The exhaustive acceptance sweep then established separate complete height and velocity zero-overlap for 902 versus 2902. Duplicate exact water fields for a pathological configured seed cause deterministic initialization failure instead of silently reducing pool diversity.

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

The public runtime ABI is version `0x00010002`. `WR_RUNTIME_ABI_CHECK()` compares the version and the compiled sizes of `WRMachine`, `WRSnapshot`, and `WRPad` before the archive can write into caller-owned objects. Rebuild the archive and Puffer binary together after any public runtime structure or contract change.

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

The configured run contains exactly 10,485,760 policy decisions, or 640 complete batches at 128 agents and horizon 128. It uses `gamma = 0.9997499687421851`, the per-0.1 s equivalent of the historical `0.9995` per-0.2 s discount. Each environment begins with a one-lap target, advances to two laps after one official finish, and advances to three after the next official finish. Curriculum state is local to each environment. The checked-in configuration enables the 128-entry authentic-wave pool described above. These configuration values preserve the prior guest-update exposure and 12.8 s recurrent horizon, but they are not training evidence by themselves.

The checked-in configuration requests at least 32 post-train evaluation episodes. Post-train evaluation reuses the existing 128-agent training vector; `base.eval_agents = 32` applies when a standalone evaluation vector is created. Before post-train evaluation, the trainer waits for rollout workers, forces every Wave Race instance to the official three-lap target, resets every environment, clears transition state, uploads the reset observations, zeros recurrent state, reinitializes action RNG state, and synchronizes the CUDA stream. Fresh CUDA evaluation, post-train evaluation, and the standalone CPU evaluator all use this same three-lap boundary. Evaluation cannot inherit one-lap curriculum state.

A fresh-process evaluation of a named checkpoint remains the acceptance method because it isolates the process lifecycle and pins the artifact under test. The checkpoint must have been trained with `OBS_SIZE = 57` and the same wave-reset contract under evaluation. OBS43 and OBS55 checkpoints have different first-layer parameter shapes and are incompatible; the CUDA and CPU loaders do not define a padding or migration rule. The selected seed-903 checkpoint below was trained from random initialization with the checked-in 128-entry authentic-wave pool. Do not use `latest`, which selects by filesystem creation time and can pick an unrelated or incompatible checkpoint:

```sh
export WR64_CHECKPOINT=/home/spark-advantage/wr64-results/obs57-authentic-waves/checkpoints/waverace64/obs57-authwave-s903-final/0000000010485760.bin
./puffer-wr64-final eval "$WR64_CHECKPOINT" --headless \
  --base.eval_deterministic=1 \
  --base.eval_episodes=128 \
  --base.eval_agents=128 \
  --base.seed=2902 \
  --env.randomize_waves=1 \
  --env.wave_seed=2902 \
  --env.wave_variants=128 \
  --env.rom_path="$WR64_ROM"
```

`puffer-wr64-final` is the retained deployment binary name. A default local `./build.sh waverace64` build is named `puffer` unless an output name is supplied.

Command-line `--section.key=value` values override [`config/waverace64.ini`](../../config/waverace64.ini). The trainer injects `train.gamma` into the environment discount, keeping the failure time cost and the optional potential-shaping modes consistent with the learner.

The adapter opts out of PufferLib's default `[-1, 1]` learner-side reward clamp. Clipping would change the configured miss, finish, and failure magnitudes. The learner therefore receives emitted rewards unchanged.

Evaluation checks the episode target after complete rollout horizons, so `base.eval_episodes` is a minimum and parallel evaluation can overshoot it. Report the actual `CUDA_EVAL games=N` denominator. For Wave Race, the machine-readable CUDA and CPU lines include deterministic or stochastic mode, exact success count, checkpoints, misses, every terminal-cause rate, target laps, three-lap success, and native successful race and lap times. A retained current result must use the native CUDA-policy binary, a fresh process, and an explicit 57-input checkpoint. `base.eval_deterministic=1` selects per-head argmax; the default samples from all five policy heads.

## Human state evaluator and capture

The interactive evaluator is a state-based visualization of the same simulation used for training. Its immutable, pointer-free capture contains the rider position, finite-difference velocity, heading and full body basis, controller state, lap and route state, native total and lap clocks, native lap splits, native physics speed, power, checkpoint and miss counts, recovery and terminal flags, the authoritative course graph and signed pass points, and a rider-centered 33 by 33 water-height grid at 128 game-unit spacing. The renderer displays that state as a low-poly water surface, course line, striped tapered left/right buoys with large diagonal pass arrows, the shared Puffer model in place of the human rider and jet ski, a chase camera, a minimap, and a compact Time Trials race HUD. The immediate target arrow blinks while the buoy body and neighboring markers remain visible. Puffer position and wave-relative bob come from the captured rider position; yaw, pitch, and roll come from the captured full body basis.

This makes the control state legible to a human, but it is not graphical parity with the N64 game. The camera, water shading, wake history, Puffer model, HUD, and minimap are renderer-owned presentation. No cartridge textures, models, framebuffer, audio, RSP output, or RDP output are used. The Puffer is the repository's existing [`resources/shared/puffer.glb`](../../resources/shared/puffer.glb), also used by Tower Climb. The 1,199,624-byte GLB has SHA-256 `6e5e201b2d08c4eae48f04d9a715ef7b5e6dbb13bffa3c6903ea656730ce7644`. This renderer uses its static bind pose and does not implement the model's morph-target animation. The edge HUD mirrors the salient Time Trials layout with `TIME`, `LAP`, `SPEED`, lap splits, `MISS`, and `POWER` or `MAX POWER`. It does not invent Championship rank or opponent portraits for the one-rider Time Trials contract. Renderer correctness means that the shown geometry and labels are projections of captured authoritative fields, within the parity boundary documented below. It does not mean pixel equivalence to Wave Race 64.

The chase camera follows horizontal rider motion directly but low-pass filters its vertical anchor with a 0.60 s time constant. Reset, discontinuity, and recovery states snap the anchor. This keeps the Puffer's wave-relative rise, fall, pitch, and roll visible instead of translating the camera eye and target by the same instantaneous rider height. The camera filter and Puffer model are evaluation-only. Training never allocates the renderer or loads the GLB, and neither changes simulator state or training observations.

Run the CUDA-policy evaluator in a visible window:

```sh
DISPLAY=:98 \
WR64_RENDER_WIDTH=640 WR64_RENDER_HEIGHT=480 \
LD_LIBRARY_PATH=/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl/lib \
./puffer-wr64-final eval \
  /home/spark-advantage/wr64-results/obs57-authentic-waves/checkpoints/waverace64/obs57-authwave-s903-final/0000000010485760.bin \
  --base.eval_deterministic=1 \
  --base.seed=16902 \
  --env.frameskip=2 \
  --env.randomize_waves=1 \
  --env.wave_seed=16902 \
  --env.wave_variants=128 \
  --policy.hidden_size=128 \
  --policy.num_layers=2 \
  --env.rom_path=/home/spark-advantage/baserom.us.rev1.z64
```

This command runs the selected randomized-wave policy on one deterministic seed-16902 variant and advances to the next authentic variant after each terminal reset. Fresh headless CUDA evaluation completed 128/128 official three-lap races on seed 2902 and another 128/128 on post-selection, training-disjoint seed 16902. The historical OBS55 seed-707 artifact cannot be loaded by the current 57-input network.

The current deployed `puffer-wr64-final` evaluator build has SHA-256 `35c7b93d3debe267ab272f84f65e4839af8d29b9d2e9863a4f999c9c96bcc99c`.

The evaluator uses one environment and advances at ten policy decisions per wall-clock second, matching two 20 Hz guest updates per action. Rendering targets 60 frames/s. Press Shift+Up once to toggle persistent HUMAN control; press the same chord again to return to POLICY. Policy inference continues behind human control so recurrent state remains synchronized for a clean handoff. The small bottom-right mode badge always shows the current owner.

| Input | Evaluator or controller action |
| --- | --- |
| Shift+Up | Toggle HUMAN/POLICY control ownership |
| W | A, throttle |
| A or Left Arrow | Full left stick X |
| D or Right Arrow | Full right stick X |
| Up Arrow | Full forward stick Y |
| Down Arrow | Full back stick Y |
| S | B, wave damping |
| Space | R, water-surface slide |
| Enter on a terminal screen | Start rendering the autoreset next race |
| Close window | End evaluation |

An official finish, disqualification, or failure freezes the actual captured terminal state. The RL core retains normal same-transition autoreset semantics, but rendered evaluation does not advance the new episode invisibly under an old finish badge. Press Enter to continue into the already-reset next race. Human/POLICY ownership persists across that restart.

For a CPU-only policy viewer or PNG capture, build the standalone evaluator under a distinct output name:

```sh
cd "$PUFFER_DIR"
WR64_DIR="$WR64_DIR" ./build.sh waverace64 wr64_eval --cpu

export WR64_CHECKPOINT=/home/spark-advantage/wr64-results/obs57-authentic-waves/checkpoints/waverace64/obs57-authwave-s903-final/0000000010485760.bin
WR64_RENDER_WIDTH=1280 WR64_RENDER_HEIGHT=720 \
  ./wr64_eval "$WR64_CHECKPOINT" \
  --base.eval_deterministic=1 \
  --base.seed=2902 \
  --env.randomize_waves=1 \
  --env.wave_seed=2902 \
  --env.wave_variants=128 \
  --env.rom_path="$WR64_ROM"
```

The default window is 800 by 450 pixels. Widths down to 480 and heights down to 270 are accepted. The compact edge HUD was verified at 640 by 360 and 640 by 480.

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

This writes `frame-000000.png` through `frame-000149.png`. At 60 rendered frames/s and ten policy decisions/s, `--capture-every=6` retains one image per policy decision and spans 15 simulated seconds. The example above retains every twelfth rendered frame, or every second policy decision, and spans 30 simulated seconds. `--capture-every=N` keeps every Nth rendered frame. `--capture-hidden` hides the window but still requires a working display and OpenGL context. `--capture-fast` removes normal render pacing and must not be used as a training-throughput measurement. Capture cannot be combined with `--headless`; `--headless` selects metrics-only CPU evaluation and does not call the renderer.

For an exact first episode from reset through its official terminal state, omit `--capture-count` and use `--capture-until-terminal`:

```sh
WR64_RENDER_WIDTH=640 WR64_RENDER_HEIGHT=480 \
  ./wr64_eval "$WR64_CHECKPOINT" \
  --env.rom_path="$WR64_ROM" \
  --base.eval_deterministic=1 \
  --capture-dir="$WR64_CAPTURE_DIR" \
  --capture-every=12 \
  --capture-hidden \
  --capture-fast \
  --capture-until-terminal
```

The terminal frame is forced into the capture even when it falls between the normal capture cadence, then the process exits before episode two. `--capture-until-terminal` and `--capture-count` are mutually exclusive.

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

The observation is a flat vector of 57 `float` values. Position, rider, course, miss, finish, recovery, water, native physics-speed, and power fields come from decompilation-derived structures and game memory. Velocity is finite-difference motion per guest update over the selected frameskip. Teleports, recovery transitions, invalid identity, and non-finite motion are excluded from the motion estimate. Any remaining non-finite feature is replaced with zero.

`WR64_SPEED_SCALE` is `55.555557` game units per guest update. Observation 55 is the cartridge's direct physics-speed field, while observation 8 remains finite-difference horizontal motion. The HUD matches the game's integer conversion by truncating native speed, multiplying that integer by `1.8`, truncating again, and capping values of 1,000 or more at 999 km/h. Power is the official integer level 0 through 5. `route_total` is the authoritative Sunny Beach route length accumulated from course-node lengths. Basis-vector names remain ordinal because the decompilation does not assign stable body-axis semantics to the three triplets.

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
| 55 | native physics speed | Cartridge physics speed field divided by `WR64_SPEED_SCALE` |
| 56 | power | Official rider power level divided by `5`; `1` is MAX POWER |

For buoy node types 0 and 1, the pass point is the node center plus a signed `400`-unit offset along the course node's decomp-derived lateral vector. Earlier adapters exposed only the node center, which withheld the required passing side from the policy.

For indices 43 through 54, let rider position be `(x, y, z)`, normalized horizontal heading be `(hx, hz)`, forward offset be `f`, and lateral offset be `l`. The sampled point and feature are:

```text
sample_x = x + f*hx + l*hz
sample_z = z + f*hz - l*hx
observation = (water_height(sample_x, sample_z) - y) * 0.01
```

The row order is forward offset `-64, 64, 192, 384`, with lateral offset `-96, 0, 96` inside each row. Units before scaling are game units. These 12 values expose nearby dynamic wave height without rendering an image or supplying the dense 33 by 33 evaluator mesh to the learner.

Changing from OBS55 to OBS57 changes the policy ABI again. Observations 0 through 54 retain their OBS55 meanings, while indices 55 and 56 expose native speed and power. Every OBS43 and OBS55 checkpoint has an incompatible first layer. There is no supported conversion or zero-padding path. Train a new model with 57 inputs.

These 57 fields are a compact control observation, not the full game state. The environment does not claim strict Markov sufficiency across every Wave Race mechanic.

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

Reward mode 0 retains strict terminal-cancelled potential shaping as an ablation. Mode 1 retains terminal potential. Both use `train.gamma` as the environment discount. Mode 2 is the current configured objective. Earlier mode comparisons used the superseded 43-observation ABI and are not learning evidence for a randomized-wave policy.

The nonterminal time cost removes a discount loophole found in pilot training. If `F` is the configured failure magnitude, the discounted sum of `-F * (1 - gamma)` on every nonterminal transition plus `-F` at a failure terminal is exactly `-F`, regardless of episode duration. Stalling until the native timeout therefore cannot make failure cheaper. A successful trajectory retains discounted base return `-F + (F + finish) * gamma^(T-1)`, so faster official finishes remain preferable.

Optional instantaneous speed and slip terms exist for experiments. The checked-in configuration keeps both coefficients at zero. Motion terms are suppressed across game-driven teleports and recovery transitions.

### Lap curriculum and evaluation boundary

Training starts each vector instance at one lap and advances that instance from 1 to 2 to 3 laps after one official success at each level. The curriculum changes only the two verified native lap-count words after restoring the selected authentic race-start state. Actions, observations, course, rider, and terminal definitions are unchanged; water-dependent physics varies by episode. Affine Lock, Bat, and Clifford provide native Puffer precedents for success-driven per-environment curricula.

Every evaluation forces three laps before reset. `target_laps=3` and equality between `three_lap_success_rate` and `success_rate` are acceptance invariants. The adapter regression checks this reset boundary, and the retained OBS57 evaluations report `target_laps=3`. Future accepted results must continue to report both fields.

## Terminal and autoreset contract

| Outcome | Condition | Reward/log classification |
| --- | --- | --- |
| Success | Official `RiderStruct.finished` flag and no disqualification | Finish reward; `success_rate` |
| Native failure | Disqualification, or generic official end without finish | Failure penalty; disqualification is logged separately from generic failure |
| Safety timeout | 14,400 guest updates without an official terminal | Failure penalty; `safety_timeout_rate` |
| Environment fault | Race identity or game state leaves the fixed contract without an official terminal | Fatal diagnostic and process abort |

On an ordinary terminal, `puf_step` records the episode log, sets terminal to `1`, restores the race-start snapshot, applies the next authentic pool variant when randomization is enabled, and returns observations from the new episode. The terminal reward and terminal flag remain in their transition buffers through autoreset. PufferLib initializes those buffers separately before the first step.

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
| `finish_time_ms` | Mean native total time over successful episodes only |
| `lap_1_ms` | Mean native first-lap split over successful episodes only |
| `lap_2_ms` | Mean native second-lap split over successful episodes only; zero for a one-lap target |
| `lap_3_ms` | Mean native third-lap split over successful episodes only; zero below a three-lap target |
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

The runtime's retained post-change isolation test initializes 16 machines, runs 512 guest updates per machine, compares serial and parallel controller streams, repeats the parallel run three times, and requires exact final RDRAM, stack, recomp context, machine state, and trajectory hashes. It passed with zero failures. The log is retained only on the deployment host at `/home/spark-advantage/wr64-recomp/runtime/isolation_acceptance_20260823T192430Z_4098593.log`; its SHA-256 is `7c910f98e02b05b2dcc4b022cdcd5b6b3103c419a673e01c1cb4b652d6faa3af`.

The adapter regression harness in [`tests/test_waverace64.cpp`](../../tests/test_waverace64.cpp) covers the 57-value ABI shape, exact placement of the 12 water features, native speed and power observations, action mapping, internal frameskip, body basis, exact time-zero reset contract, guest halfword lane, buoy side, lap-wrap continuity, official misses and disqualification, discount-correct failure shaping, shortened one-lap official finish, full three-lap scripted reachability, B/stick-Y interventions, observation ranges, deterministic baselines, authentic wave-pool reset behavior, and vector affinity/ownership. The pool regressions prove exact permutation replay, complete K-step cycling without duplicates, distinct water observations, divergent authoritative trajectories under identical controls after normalizing unrelated primary RNG state, and ordinary cartridge water evolution without advancing the host variant index. Native donor and reconstructed owner agree on reset RDRAM, adapter state, observations, reward buffers, and render projection. Randomized vectors at N=2, 4, and 8 with K=4 prove canonical snapshot rebasing, curriculum application, shared-pool lifetime after closing environment 0, and absence of cross-environment corruption. The unit harness does not invoke the central CUDA train/evaluation path.

The checked-in exact-water test compares `wr64_water_height` bit-for-bit against the recompiled cartridge function `func_8004D30C` at 4,096 deterministic points on the live water field. It exercises both interpolation branches, invokes the reference on scratch RDRAM because the recompiled function uses a guest stack, and verifies that the pure sampler does not change live RDRAM. The implementation was also characterized at 262,144 points across 64 randomized complete water fields with zero float-bit mismatches when compiled with `-ffp-contract=off`.

That result is exact equivalence to the recompiled `func_8004D30C`, not an independent emulator comparison of wave evolution. It validates the query used by observations 43 through 54 and the evaluator mesh. It does not extend the held-A interpreter parity trace to other wave states or inputs.

Dynamic water is causal simulator state, not renderer decoration. Flattening the live water field while holding the initial machine state and controls fixed changed the authoritative trajectory by the second guest update. In an unmodified 20-s trace, rider Y covered 24.066 game units, local water height covered 21.840 units, wave-relative clearance covered 28.292 units, pitch ranged from about -1.6 to +11.2 degrees, and roll ranged from about -11.5 to +5.8 degrees. The vertical camera filter makes part of that motion visible without changing the physics.

The render-state regression captures the state twice, requires byte-identical logical content and hashes, checks every course pass point, checks every value in the 33 by 33 water tile bit-for-bit against `wr64_water_height`, verifies live RDRAM is unchanged, and verifies that capture alone leaves `env->client` null. It validates the state projection, not Raylib raster output or N64 visual parity.

The separate renderer regression in [`tests/test_waverace64_render.cpp`](../../tests/test_waverace64_render.cpp) passed on an isolated X display using the tested software-rendered stack:

| Check | Retained result |
| --- | --- |
| Training path stays lazy | `Client` remained null through 192 decisions, including autoresets |
| Control ownership | Pure rising-edge test and real X11 injection both produced POLICY to HUMAN to POLICY from two Shift+Up chords |
| Pure authoritative capture | State hash `f2b411ab54afcf26`, including native clocks, speed, and power |
| Renderer preserves simulator core | Eight repeated `puf_render` calls left RDRAM, stack, recomp context, machine scalars, adapter state, and TLS owner unchanged |
| Fixed-state pixels and model lifecycle | Two captures were byte-identical; pixel hash `25cf8bb430bd8f98`; the shared Puffer GLB loaded as a valid model and unloaded cleanly |
| Render cadence independence | Headless and rendered 192-decision trajectories matched; hash `bb75cd7a071f6551` |
| Wave-relative Puffer motion | A 48-frame, 4.8-s simulated trace exposed 12.055 pixels of exact teal-body centroid movement and 21 pixels of silhouette-bottom movement while preserving simulator state |
| Moving capture | 48 rendered frames reached guest-update tick 96; final state hash `3217579d69426366` |

The pixel hash is a regression baseline for the tested Raylib/OpenGL software stack. It is not asserted to be portable across graphics drivers, and it is not a reference N64 framebuffer hash.

### Adapter acceptance and remaining parity gaps

| Gate | Status |
| --- | --- |
| Shortened one-lap official finish fixture | **PASS:** official finish at update 1,758, native time 87,835 ms, and action hash `64157b7ea07f2a23`. |
| Unmodified three-lap scripted reachability through the public observation interface | **PASS:** 1,300 decisions, 5,200 updates, zero misses, native time 259,780 ms, splits 79,922/92,115/87,743 ms, and action hash `6c5a285ee76ce8e0`. This is controller reachability, not policy learning. |
| B and stick-Y authoritative interventions | **PASS:** B and stick Y each changed the authoritative trace in all 6 probes, including recovery. |
| Generic Puffer regression | **PASS:** stock CartPole CPU evaluation repeated byte-for-byte at fixed seed; stock Breakout CUDA build and evaluation exited zero. |
| Multi-trace interpreter parity, including B and R action regimes | **PENDING** |
| Post-reset secondary RNG and whole-RDRAM parity | **PENDING** |

The deterministic controller fixtures establish task reachability and regression behavior. They do not establish policy learning or current training performance.

### Authentic-wave pool acceptance

The retained all-variant acceptance sweep built the 128-entry seed-902 pool, then booted all 128 donors independently and sequentially. Every reconstructed owner matched its native donor on the full 8 MiB reset RDRAM, adapter `State`, 57 observations, transition buffers, render projection, and host-visible machine scalars. Two deterministic 32-decision scripts were run from a fresh reset for every variant; full RDRAM and host-visible state matched after every decision. This establishes the state-transplant contract for all checked pool entries. Native stacks and recompilation contexts remain owner-specific by design and are not claimed byte-identical to donor objects.

The same sweep found 128 unique full-RDRAM states, complete water fields, height fields, and velocity fields for seed 902. A separate 128-entry pool at held-out seed 2902 also had 128 unique entries. The two pools had zero overlap in boot-time values, full-RDRAM hashes, complete water fields, height fields, and velocity fields. The 1,024-reset stress test replayed exactly, visited all 128 fields, and reported zero recoveries, faults, or terminals during the first 20 fixed-A guest updates. Maximum absolute vertical velocity was 2.384010 game units/update and maximum vertical displacement was 8.785711 game units.

An exact-byte follow-up rejected seed 3902 as disjoint-pool evidence. Training variant 55 equals seed-3902 variant 8, and training variant 114 equals seed-3902 variant 100, in complete water, heights, and velocities. Cross-pool counts were boot ticks 0, full RDRAM 0, complete water 2, heights 2, and velocities 2. The earlier seed-3902 policy evaluation remains valid playback evidence, but it is not disjoint-pool evidence. Post-selection replacement seed 16902 has 128 internally unique complete-water fields and zero overlap with training seed 902 in all five categories.

The exact acceptance source and output are retained under `/home/spark-advantage/wr64-results/obs57-authentic-waves/audit`. `probe_wave_pool_acceptance.cpp` has SHA-256 `fbac23774ddd8f7d01bb80367f27828d03e2cbabb51df50d0a3ac58983533586`; `wave-pool-acceptance-final.log` has SHA-256 `8d9d3295ef4bf9627aa533142ef825d36cce6a5d8770b098b059a762fdf0e694`. The exact overlap probe source has SHA-256 `6dfff49032703d81867b68b364f65beadc67a587dbf8f5e9d81f14796b982851`; its seed-3902 rejection and seed-16902 acceptance logs have hashes `b41f8317fb36e003d2e0736f44be6abbfa0e3056c89addaa9b598769de87abe3` and `d9219624668dcb17e5eaa93f91ce1a5896ce899cde3f91cc6291edff9511216d`. The stress source hash is `41617a89dea347118dbb1906c372bf9927067c015b6113cf80bf77978a6c16d6` and its log hash is `db7792dc4b144d9482a163b637ffb08231b1180b46e9eebb1c1f58e5925265e5`. The tick-schedule source hash is `a5cf0194c678df14bf788d938c8180829d091d295bb9f40447bc05b2c2330b52` and its log hash is `3d7b4d971e8195767b23fbd1f9f7f593469203afe43883826edbccb82bed8497`. Final runtime ABI/environment, adapter, native-trainer build, renderer, and renderer-UBSan log hashes are `0073b21b657007a8db38bc4b9faef269b3af2d969eb58492f5a83d94dd856618`, `9ce38dde8259e3834941ea59f954e2aa0737a7a7b536c81f9a9750bcee9ae52e`, `2f8bd52527094f9b0037c1eed2934a020e17f96f1579378f8c2c35caa5994e3c`, `49fce4c37b35771c297e04d736833b0b3f94a3fda2312453cb8416f2fe7a4f9e`, and `49fce4c37b35771c297e04d736833b0b3f94a3fda2312453cb8416f2fe7a4f9e`. `SHA256SUMS-final`, itself SHA-256 `04bb8d9ba04dbb02f1e7732c4cd96626c8b8f8faf2a4cd6b51dcf428b626ff65`, covers every retained source, log, and renderer image in that directory.

Under one identical renderer-free adapter rollout protocol with 128 environments and 16 OpenMP threads, fixed-wave trials measured 55,380, 57,135, and 58,950 decisions/s; K=128 randomized trials measured 56,315, 28,184, and 57,529 decisions/s. The randomized median was 56,315 decisions/s versus 57,135 fixed, a 1.44% reduction. The low randomized sample is retained rather than discarded. K=128 initialization measured 8.647 s and 106,968 KiB maximum RSS for N=1, and about 3.4 s and 1,893,084 KiB for N=128. This protocol isolates simulator rollout and pool lifetime; it is not a complete CUDA trainer measurement. The benchmark source SHA-256 is `d05569e25c83d85e4dd166ca2a25ac684ba92a97d6f3654c334f7ee5cc784392`; its retained log SHA-256 is `6c6a370ab5b21294bd0ca3e9162f232f7f63394efb5ecdf526d8a0bd753c36c7`.

## Performance and learning acceptance

### Authentic-wave OBS57 evidence

Two renderer-free PufferLib 5.0 runs were trained from random initialization on the seed-902, K=128 authentic-wave pool. Each completed 10,485,760 decisions. Seed 902 took 186.54 s wall time, or 56,211.86 decisions/s. Seed 903 took 185.34 s, or 56,575.81 decisions/s and 113,151.62 native guest updates/s, with 1,683% CPU and 1,688,236 KiB maximum RSS. The final seed-903 training batch reported 100% official three-lap success, no failure terminal, and `target_laps=3`. Training used the host-CPU recompiled simulator; CUDA ran policy inference, recurrent state, rollout tensors, and PPO learning. The renderer and Raylib were inactive.

An identical 2,097,152-decision paired trainer benchmark took 43.81 s with fixed resets and 43.02 s with K=128 randomized resets. The randomized run was 1.8% faster in this pair, so reset-time page-delta selection introduced no measurable trainer penalty. The pair measures system throughput only and does not establish an intrinsic randomization speedup.

The selected checkpoint is:

| Path | Size | Parameters | SHA-256 |
| --- | ---: | ---: | --- |
| `/home/spark-advantage/wr64-results/obs57-authentic-waves/checkpoints/waverace64/obs57-authwave-s903-final/0000000010485760.bin` | 438,272 bytes | 109,568 FP32 | `91217a7eb1ff5f4d553b678c206c5acaa727461bcf45cfd4f0266f7c2e0f62bf` |

Fresh CUDA processes evaluated held-out wave seed 2902 and post-selection, training-disjoint seed 16902. Seed 16902 was separately shown internally unique and disjoint from training seed 902 in boot ticks, full RDRAM, complete water, heights, and velocities. Every evaluation forced the official three-lap target. The deterministic policy completed all 128 episodes on each pool. Stochastic sampling completed 510/512 on each pool, or 99.6094%. Both stochastic runs had one generic failure and one disqualification; all four runs had zero safety timeout and zero environment fault.

The deterministic result was repeated with one environment for 128 episodes on both seed 2902 and seed 16902. Each odd-stride reset schedule completed one exact permutation of all 128 variants, avoiding any ambiguity from parallel autoreset accounting. Both finished 128/128 with zero adverse terminal. The retained seed-2902 and seed-16902 log SHA-256 values are `554e15b1b016cba7db53bac54bb2adb46a9aa2d3f93877b41c3ec95a7cd668f7` and `3929c1e30c8fbdb0b577575aec71ef4230851e5d568103a5b29620c6f92ffc42`.

| Wave seed | Action mode | Episodes | Successes | Success rate | Mean misses | Generic failures | Disqualifications |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2902 | Per-head argmax | 128 | 128 | 1.000000 | 0.992188 | 0 | 0 |
| 2902 | Stochastic | 512 | 510 | 0.996094 | 0.560547 | 1 | 1 |
| 16902 | Per-head argmax | 128 | 128 | 1.000000 | 0.828125 | 0 | 0 |
| 16902 | Stochastic | 512 | 510 | 0.996094 | 0.564453 | 1 | 1 |

The seed-2902 deterministic and stochastic log SHA-256 values are `94e7143f5f2a42f261ab33527c72690482f03607fc1e1d8c643829c925f4fecb` and `ee6dff8037fe3c030d8dcc7890dc257add4b3923d5718109a8b4a039217b6f03`. The seed-16902 values are `32d52753f082d7f670baa4284fb702a59582ffc58d5c7c27702614d676732163` and `a444de2d01c74c8774a19b4f929788f38ac65306ebbccc0390fbf28550966201`. Seed 2902 was constructed and exhaustively compared with the training pool before policy selection. Seed 16902 was screened and evaluated only after seed 903 had been selected.

A fresh standalone CPU process loaded exactly 109,568 floats from the selected checkpoint and completed one deterministic seed-16902 official race in 883 policy decisions. It cleared 46 route checkpoints with one miss and zero failures, disqualifications, timeouts, or faults. The CPU evaluator SHA-256 is `fb393420cd494ada8059ef911bead077f061a2520498bc2f499ae661ada85624`; the retained CPU log SHA-256 is `ad79bb3f262092c30d893ac4bad1f441c54855a3c1d402e840eb439117831f16`.

The terminal-aware CPU evaluator ran the selected checkpoint and retained 5,294 960 by 540 source frames from the exact seed-16902 time-zero state through the official finish at 87.898 s. This verifies learned-policy playback and renderer behavior without asserting bit-identical CPU and CUDA inference. The encoded H.264 High/yuv420p MP4 runs at 60 frames/s, contains 5,534 decoded frames after a 240-frame frozen-finish tail, lasts 92.23 s, is 21,310,915 bytes, and has SHA-256 `a89e81fbb42be0212c3c175b4621efe3fde957e753f59394989e8ab22f6a5835`. Full decode completed without error. Independent inspection of start, live-race, midpoint, source-terminal, and decoded-tail frames confirmed visible dynamic wave geometry, Puffer bob and orientation, directional buoy arrows, all three laps, and the official finish with no episode-two contamination. Capture, encode, and full-decode log SHA-256 values are `4ff46be7f4deeb0319463dea455c2d29a3c05a4737a1530db7fc4861bd340397`, `583d73abff232f75b1b0415ffca7e9822071ae6f5baa3cfdea24a790c25b27da`, and `a5b633665078c0c6bfa4a2fc49fa2162a4df6e9684a3a7f386f9cb8a3909204f`.

### Historical fixed-wave OBS57 evidence

The measurements, checkpoint, evaluations, and recordings in this subsection predate authentic per-episode wave-pool selection. They remain pinned provenance for the fixed-wave OBS57 contract, but they are not learning acceptance for the checked-in randomized-wave configuration.

The clean frameskip-2 seed-900 run completed the configured 10,485,760 decisions in 201.845 s of trainer uptime. That is 51,949.565260 policy decisions/s and 103,899.130521 native game updates/s. The run used the renderer-free training path. The binary used for this retained measurement has SHA-256 `f20a62342714f2c0698d356435849d0f3069fbed703e3e7c83ba510c15cff9d2`; the current post-measurement Puffer evaluator build is identified above.

The selected seed-901 checkpoint is:

| Path | Size | SHA-256 |
| --- | ---: | --- |
| `/home/spark-advantage/wr64-results/obs57-seeds/checkpoints/waverace64/obs57-fs2-s901/0000000010485760.bin` | 438,272 bytes | `eaf2d9be637f5d03a95bb1d6ff9c40096867e977dfcdbd1eb8f94df855f277b5` |

Fresh CUDA evaluation produced:

| Action mode | Episodes | Successes | Success rate | Misses | Failures | Disqualifications | Finish time | Lap splits | Episode updates | Mean speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Per-head argmax, seed 1901 | 128 | 128 | 1.000000 | 0 | 0 | 0 | 82,607 ms | 29,643/26,029/26,935 ms | 1,654 | 1,018.898 game units/s |
| Stochastic, seed 3901 | 512 | 509 | 0.994141 | 0.330078 mean | 2 | 1 | 84,200.930 ms | 28,945.953/27,441.498/27,813.461 ms | 1,683.535 mean | 1,040.705 game units/s |

Both evaluations forced `target_laps=3`. Deterministic evaluation reported no safety timeouts or environment faults. Held-out stochastic evaluation reported no safety timeouts or environment faults; its finish and lap times are conditioned on successful episodes.

A fresh standalone CPU evaluator built from the same committed OBS57 source loaded all 109,568 checkpoint floats and completed one deterministic official three-lap episode in 784 policy decisions. It passed all 48 checkpoints with zero misses, failures, disqualifications, safety timeouts, or environment faults. The deployed `wr64_cpu` binary has SHA-256 `62f8a3be97fda9659f204e4a6e08aac09b57aaa3ded2e19c31b89fc2116c73ab`. Its retained log is `/home/spark-advantage/wr64-results/obs57-seeds/evals/final-cpu-puffer-det-s1901.log`, SHA-256 `b31035bc62f6386a72ca2a28c8ad26c034d633be57473bceedcee55bcdbe5b44`. This is a checkpoint-format and official-completion smoke test, not a claim of bit-identical CPU and CUDA policy inference.

### Historical OBS55 evidence

The following measurements and artifacts belong to the superseded 55-input, frameskip-4 contract. They remain useful provenance, but they are not compatible checkpoints or current-contract acceptance evidence. Two complete renderer-free runs were measured:

| Training seed | Decisions | Trainer uptime | Decisions/s | Guest updates/s | Process wall | Wall decisions/s | CPU | Maximum RSS | Internal GPU mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 707 | 5,242,880 | 170.346 s | 30,777.828655 | 123,111.314618 | 171.28 s | 30,609.995329 | 1,687% | 1,625,884 KiB | Not retained |
| 708 | 5,242,880 | 170.704 s | 30,713.28 | 122,853.13 | 171.65 s | 30,544.01 | 1,687% | 1,625,936 KiB | 4.287% |

Historical seed 707 ran `./puffer-wr64-obs55`, SHA-256 `893c63d0cde3fb9d8073232fe5b1c98a543f53e00732c01b5488e7c7817952f`. Historical seed 708 ran `./puffer-wr64-final`, SHA-256 `e96263824d6083e08f8d18d57aa415dcf68c2ca2a9260d1b8192cbf90a59633c`. The binaries remain distinct provenance artifacts for that superseded contract.

The seed-708 GPU value is the time-weighted internal metric for that run. No equivalent full-run value is retained for seed 707. The renderer and capture paths were inactive in both runs. Seed 707's final logged curriculum batch reported success 1.0 and `target_laps=3`; this is a final-batch training result, not an evaluation statistic.

The retained 43-input historical baseline had mean 30,900.2 and median 30,914.5 policy decisions/s. It is not documented as a fully pinned duration-matched comparison with the OBS55 runs, so no regression percentage is assigned. The OBS55 network had 109,312 parameters versus 107,776 for OBS43, an increase of 1,536 or 1.425%. The evidence does not isolate water-query cost from network, host, placement, or run-protocol effects.

Three earlier 55-input short screens averaged 29,609.44 decisions/s, 16.220 CPU equivalents, 1.55052 GiB mean peak RSS, and 4.384% device-wide GPU utilization. They are retained as preliminary screens, not substituted for the complete runs. The GPU figure is device-wide and was not sampled continuously or attributed to this process. Their provenance is deployed Puffer commit `8a9c51f2cebac95c36119d2eb975d53db10b4529`, local renderer-harness commit `1fa4769b942a6179828a2e7b5d5773e8d3596234`, runtime commit `e3f56302898a98ec7f7b20ca35fc1b5de69fe890`, trainer SHA-256 `893c63d0cde3fb9d8073232fe5b1c98a543f53e00732c01b5488e7c7817952f`, and CPU evaluator SHA-256 `a07089133be8f5c32ef89119313f6702170a98d2566451c82f0b56467a0d1161`.

Both historical OBS55 checkpoints are 437,248 bytes and incompatible with OBS57:

| Training seed | Checkpoint path | SHA-256 |
| ---: | --- | --- |
| 707 | `/home/spark-advantage/wr64-results/state-eval-obs55-e0a8e2d4/checkpoints/waverace64/state-eval-obs55-s707/0000000005242880.bin` | `a6696e3888ca472712071aa9fd6b82b377e3ddf956db41ca2082488c3145fc59` |
| 708 | `/home/spark-advantage/wr64-results/state-eval-obs55-seed708-final/checkpoints/waverace64/state-eval-obs55-s708/0000000005242880.bin` | `83382a3f31141a1645a5be2cb8c31696480066838faaee70c8527138722027dd` |

Historical fresh-process evaluation produced:

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

The historical terminal-aware CPU capture retained the reset frame, every one of those 619 deterministic policy decisions, and the official terminal frame as 620 source PNGs. It stopped before any frame from episode two. A separate historical CUDA-policy recording used the frozen finish screen as a synchronization gate, restarted once, and retained the first rendered policy transition through 120 consecutive frames of the next official finish screen. That OBS55 H.264 High/yuv420p MP4 is 640 by 360 at 60 frames/s, contains 6,506 decoded frames, lasts 108.433 s, is 11,085,404 bytes, and has SHA-256 `041bba61d8f86a4a00e4aec555a9b71a7d47e287696aaba7cc60823f31dbfc1b`. Its final state reports lap 3/3, 46 cleared gates, zero misses, and official finish at 106.60 s.

The fixed-wave native CUDA-policy Puffer recording is retained at `/home/spark-advantage/wr64-results/obs57-seeds/video/waverace64-obs57-s901-puffer-full-race-20260823.mp4`. It is H.264 High/yuv420p at 960 by 540 and 60 frames/s, contains 5,348 decoded frames, lasts 89.133 s, is 23,783,474 bytes, and has SHA-256 `42e3db2597955bbc43d5078735635afa590c9dcc38fc316cf481640d12a12b60`. The file begins at the exact native time-zero state, contains the complete deterministic 82.607 s official race, and ends with more than 6 s of the frozen official finish. Frame inspection at 0, 10, 82, and 88 s verifies the time-zero start, live Puffer wave motion, final lap, terminal overlay, and absence of a captured X11 cursor or episode-two contamination. The prior rider-and-jet-ski OBS57 recording remains a historical pre-Puffer artifact.

| Fixed-wave OBS57 provenance item | Status |
| --- | --- |
| Full renderer-free training throughput | **MEASURED:** clean seed 900 reached 51,949.565260 decisions/s and 103,899.130521 native updates/s |
| Fixed-wave checkpoint | **RETAINED:** selected seed-901 OBS57 checkpoint, 438,272 bytes, with hash above |
| Fresh deterministic CUDA three-lap evaluation | **PASS:** 128/128, zero misses and no failure terminal; native finish 82,607 ms |
| Fresh stochastic CUDA three-lap evaluation | **PASS:** 509/512 (`0.994141`) under held-out action seed 3901 |
| Native time reporting | **PASS:** deterministic splits 29,643/26,029/26,935 ms sum exactly to 82,607 ms |
| Human evaluator static regressions | **PASS:** valid shared Puffer model, compact Time Trials HUD, native clocks/speed/power, wave-relative motion, terminal freeze, and two-way Shift+Up toggle |
| Fixed-wave Puffer full-race MP4 | **PASS:** selected OBS57 CUDA policy from native time zero through the frozen official finish; cursor-free, 89.133 s, 5,348 frames, hash documented above |
| Broader OBS57 training-seed statistics | **PENDING:** the selected seed and held-out action evaluation do not estimate the wider training-seed distribution |

## Audit

[`AUDIT.md`](AUDIT.md) traces the prior claims, the evidence behind each verdict, the repairs already made, and the remaining CUDA, performance, and learning gaps.
