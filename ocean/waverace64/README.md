# Wave Race 64 headless training environment

This environment integrates a statically recompiled Wave Race 64 US Rev 1 core with the native PufferLib 5.0 trainer. The game simulation executes as native host CPU code. PufferLib runs policy inference, rollout tensors, and learning on CUDA. Every environment instance owns its guest memory and suspended game context.

The supported task is deliberately narrow: rider 0, one rider, Sunny Beach, Time Trials, three laps, and one deterministic race-start snapshot. Rendering, audio, RSP/RDP work, controller-pak behavior, multiplayer, other riders, other courses, and other modes are outside the environment contract.

## Status against the original assignment

The original assignment required a native Wave Race 64 core in the current PufferLib 5.0 stack, high training throughput, demonstrated parity with the game, and preferably a CUDA simulator that avoids CPU simulation work.

| Requirement | Current status |
| --- | --- |
| Native Wave Race 64 core | Implemented with 1,203 statically recompiled game functions and a headless libultra runtime. |
| Current PufferLib 5.0 integration | Implemented against upstream `5.0` commit `ba238f8c` using the native C++17/CUDA trainer. |
| Parity with the game | Proven for selected authoritative state on one pinned deterministic interpreter trace through a native failure terminal. Broader parity remains unproven. |
| High-throughput training | **PENDING:** final clean-worktree benchmark matrix and profiler breakdown. |
| Learning quality | **PENDING:** production three-lap convergence and checkpoint evaluation. |
| CUDA, CPU-free simulation | Unimplemented. The policy and learner use CUDA; `libwr64.a` executes on host CPU workers. |

## Execution architecture

```text
CUDA policy and learner
        | actions and rollout tensors
        v
PufferLib CPU vector workers
        | five controller heads, 43 observations, reward, terminal
        v
waverace64.h adapter
        | WRPad and public runtime ABI
        v
libwr64.a on the host CPU
        | native recompiled game code and headless OS shims
        v
per-instance 8 MiB RDRAM, native stack, ucontext, and snapshot
```

Training is unpaced. The guest's simulated-time cadence does not limit wall-clock throughput. GPU utilization cannot remove the CPU simulator cost in this architecture.

The adapter follows the native Puffer environment interface in [`waverace64.h`](waverace64.h): `puf_init`, `puf_reset`, `puf_step`, `puf_close`, `puf_log`, and `puf_render`. `puf_render` is an intentional no-op. The ASCII utility in the runtime repository is a telemetry plot and is not a renderer.

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

Train with the production configuration:

```sh
cd "$PUFFER_DIR"
./puffer train --env.rom_path="$WR64_ROM"
```

The production configuration requests at least 32 post-train evaluation episodes. Post-train evaluation reuses the existing 128-agent training vector; `base.eval_agents = 32` applies when a standalone evaluation vector is created. Before post-train evaluation, the trainer waits for rollout workers, resets every environment, clears transition state, uploads the reset observations, zeros recurrent state, reinitializes action RNG state, and synchronizes the CUDA stream. Post-train metrics therefore do not begin with partial training episodes or recurrent state from training.

A fresh-process evaluation of a named checkpoint remains the retained acceptance method because it isolates the process lifecycle and pins the artifact under test. Use an explicit path because `latest` selects by filesystem creation time and can pick an unrelated benchmark checkpoint:

```sh
export WR64_CHECKPOINT=/path/to/checkpoints/waverace64/run/step.bin
./puffer eval "$WR64_CHECKPOINT" --headless \
  --base.eval_episodes=100 \
  --base.eval_agents=128 \
  --env.rom_path="$WR64_ROM"
```

Command-line `--section.key=value` values override [`config/waverace64.ini`](../../config/waverace64.ini). The trainer injects `train.gamma` into the environment discount, so a gamma override automatically preserves the potential-shaping identity.

Evaluation checks the episode target after complete rollout horizons, so `base.eval_episodes` is a minimum and parallel evaluation can overshoot it. Report the actual `CUDA_EVAL games=N` denominator. The standalone `--cpu` evaluator is excluded from this procedure until its GCC-LTO archive link is independently retained as passing.

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

The observation is a flat vector of 43 `float` values. Position, rider, course, miss, finish, and recovery fields come from decompilation-derived structures. Velocity is finite-difference motion per guest update over the selected frameskip. Teleports, recovery transitions, invalid identity, and non-finite motion are excluded from the motion estimate. Any remaining non-finite feature is replaced with zero.

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
| 11 | reserved Z | Always `0` in the production five-head action contract |
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

For buoy node types 0 and 1, the pass point is the node center plus a signed `400`-unit offset along the course node's decomp-derived lateral vector. Earlier adapters exposed only the node center, which withheld the required passing side from the policy.

These 43 fields are a compact control observation, not the full game state. The environment does not claim strict Markov sufficiency across every Wave Race mechanic.

## Reward

Production coefficients are:

| Term | Coefficient |
| --- | ---: |
| Speed | `0` |
| Route progress | `10` per accumulated lap of route distance |
| Slip | `0` |
| Successful checkpoint | `1` |
| Miss | `-5` per official miss event |
| Official finish | `+100` |
| Failure | `-20` |

Progress and checkpoint shaping use a discount-correct potential:

```text
Phi(s) = reward_progress * progress_total / route_total
       + reward_checkpoint * checkpoints

F(s, s_next) = gamma * Phi(s_next) - Phi(s)    on a nonterminal transition
F(s, terminal) = -Phi(s)                       on a terminal transition
```

`train.gamma` is `0.999` in production and is also the environment's potential discount. Gamma is applied per policy transition, where each transition contains four guest updates. This construction telescopes under the learner's discount and prevents an agent from banking progress shaping before a later failure. Miss, finish, and failure terms remain outside the potential.

Optional instantaneous speed and slip terms exist for experiments. Production keeps both coefficients at zero. Motion terms are suppressed across game-driven teleports and recovery transitions.

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

The runtime's retained isolation test initializes 16 machines, compares serial and parallel controller streams, repeats the parallel run three times, and requires exact final RDRAM, stack, recomp context, machine state, and trajectory hashes. The retained hardened run is `runtime/isolation_acceptance_20260822T050325Z_3003381.log`; its SHA-256 is `805b8384d97306f77c043734727a5cd0ffbece35088574bbb4ebf9ba165d852f`.

The adapter regression harness in [`tests/test_waverace64.cpp`](../../tests/test_waverace64.cpp) covers the ABI shape, action mapping, internal frameskip, body basis, reset contract, guest halfword lane, buoy side, lap-wrap continuity, official misses and disqualification, discount-correct failure shaping, shortened one-lap official finish, full production three-lap finish, B/stick-Y interventions, observation ranges, deterministic baselines, and vector affinity/ownership.

### Adapter acceptance and remaining parity gaps

| Gate | Status |
| --- | --- |
| Shortened one-lap official finish fixture | **PASS:** official finish at update 1,070 with action hash `c6ae00920fd86802`. |
| Unmodified production three-lap finish using only the 43 observations | **PASS:** frameskip 4, 2,334 decisions, 9,336 guest updates, score 87,582.3, zero misses, and official success. |
| B and stick-Y authoritative interventions | **PASS:** B changed the authoritative trace in 5 of 6 probes; stick Y changed it in all 6. |
| Multi-trace interpreter parity, including B and R action regimes | **PENDING** |
| Post-reset secondary RNG and whole-RDRAM parity | **PENDING** |

The retained deterministic rerun exited zero in 4.7 s. Its no-op baseline ended after 2,997 decisions with `perf = 0`. Three fixed-seed random baselines ended after 85, 90, and 85 decisions with `perf = 0.0639`, `0.0556`, and `0.0293`; the harness requires zero official successes for these baselines. These controller fixtures establish task reachability and regression behavior. They do not establish policy learning.

## Performance and learning acceptance

No final number is recorded until it is measured on the repaired upstream 5.0 port with the exact production ROM, 43 observations, five action heads, frameskip 4, and current runtime archive.

| Performance result | Required reporting | Status |
| --- | --- | --- |
| Policy decisions per second | Wall time, agents, buffers, worker threads, horizon, and frameskip | **PENDING** |
| Guest game updates per second | Decisions per second multiplied by actual updates per decision | **PENDING** |
| CPU utilization and scaling | Per-core utilization plus 1/8/16/20-thread sweep | **PENDING** |
| GPU utilization | Device, utilization, policy time, copy time, and learner time | **PENDING** |
| Memory scaling | Resident memory at representative vector sizes | **PENDING** |
| Affinity robustness | Bound and unbound runs with identical rollout worker availability | **PENDING** |

The minimum useful matrix should compare frameskip 1 and 4, agents 128/256/512, threads 16/20, and buffers 1/2/4 where memory permits. Report policy decisions per second and guest updates per second separately.

| Learning result | Acceptance criterion | Status |
| --- | --- | --- |
| No-op and random baselines | Zero production success with retained `perf`, miss, and terminal-cause data | **PASS:** deterministic harness values retained above |
| Five-million-transition production run | Complete run with checkpoints, misses, terminal causes, return, and SPS | **PENDING** |
| Checkpoint evaluation | Headless production evaluation against baselines | **PENDING** |
| Seed sensitivity | At least three training seeds with the same evaluation protocol | **PENDING** |
| Three-lap learning | Demonstrated nonzero official `success_rate` on unmodified three-lap episodes | **PENDING** |

A compile, a short rollout, or a shortened one-lap scripted finish does not establish learning quality.

## Audit

[`AUDIT.md`](AUDIT.md) traces the prior claims, the evidence behind each verdict, the repairs already made, and the remaining CUDA, performance, and learning gaps.
