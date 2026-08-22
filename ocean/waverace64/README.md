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
| High-throughput training | Measured at a five-seed median of 30,914.5 policy decisions/s and 123,658.1 guest updates/s. Simulation remains CPU-bound. |
| Learning quality | Official three-lap finishes learned in independent production runs and verified in fresh-process stochastic and argmax evaluation. Seed sensitivity is material and reported below. |
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

The production run contains exactly 5,242,880 policy decisions, or 640 complete batches at 128 agents and horizon 64. Each environment begins with a one-lap target, advances to two laps after one official finish, and advances to three after the next official finish. Curriculum state is local to each environment.

The production configuration requests at least 32 post-train evaluation episodes. Post-train evaluation reuses the existing 128-agent training vector; `base.eval_agents = 32` applies when a standalone evaluation vector is created. Before post-train evaluation, the trainer waits for rollout workers, forces every Wave Race instance to the official three-lap target, resets every environment, clears transition state, uploads the reset observations, zeros recurrent state, reinitializes action RNG state, and synchronizes the CUDA stream. Fresh CUDA evaluation, post-train evaluation, and the standalone CPU evaluator all use this same three-lap boundary. Evaluation cannot inherit one-lap curriculum state.

A fresh-process evaluation of a named checkpoint remains the retained acceptance method because it isolates the process lifecycle and pins the artifact under test. Use an explicit path because `latest` selects by filesystem creation time and can pick an unrelated benchmark checkpoint:

```sh
export WR64_CHECKPOINT=/path/to/checkpoints/waverace64/run/step.bin
./puffer eval "$WR64_CHECKPOINT" --headless \
  --base.eval_episodes=100 \
  --base.eval_agents=128 \
  --env.rom_path="$WR64_ROM"
```

Command-line `--section.key=value` values override [`config/waverace64.ini`](../../config/waverace64.ini). The trainer injects `train.gamma` into the environment discount, keeping the failure time cost and the optional potential-shaping modes consistent with the learner.

The adapter opts out of PufferLib's default `[-1, 1]` learner-side reward clamp. Clipping would change the configured miss, finish, and failure magnitudes. The learner therefore receives emitted rewards unchanged.

Evaluation checks the episode target after complete rollout horizons, so `base.eval_episodes` is a minimum and parallel evaluation can overshoot it. Report the actual `CUDA_EVAL games=N` denominator. For Wave Race, the machine-readable CUDA and CPU lines include deterministic or stochastic mode, exact success count, checkpoints, misses, every terminal-cause rate, target laps, and three-lap success. Retained production evaluation uses the native CUDA-policy binary, a fresh process, and an explicit checkpoint. `base.eval_deterministic=1` selects per-head argmax; the default samples from all five policy heads.

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
| New route frontier | `3` per lap of newly reached route distance |
| Slip | `0` |
| Successful checkpoint | `0.3` |
| Miss | `-0.5` per official miss event |
| Nonterminal time cost | `-reward_fail * (1 - gamma)` per policy transition |
| Official finish | `+10` |
| Failure | `-2` |

Production uses reward mode 2. It credits only a new maximum route frontier and verified checkpoint events:

```text
frontier_gain = max(0, max_progress_after - max_progress_before)
shaping = 3 * frontier_gain / route_total
        + 0.3 * successful_checkpoint_events
```

The maximum frontier is monotone, so reversing and revisiting a segment cannot earn it twice. A route-node advance accompanied by an official miss earns no checkpoint term. Recovery, teleport, invalid-identity, and discontinuous route transitions earn neither motion nor frontier credit. This matches the practical Puffer racing pattern of dense distance progress plus discrete gates while adding a one-credit frontier guard against oscillation.

Reward mode 0 retains strict terminal-cancelled potential shaping as an ablation. Mode 1 retains terminal potential. Both use `train.gamma` as the environment discount. A 5,242,880-decision mode-0 pilot learned no official finish and evaluated at `perf=0.0266`; its failure returns were mathematically valid but gave successful partial navigation no lasting task credit. Mode 2 is the measured production objective.

The nonterminal time cost removes a discount loophole found in pilot training. If `F` is the configured failure magnitude, the discounted sum of `-F * (1 - gamma)` on every nonterminal transition plus `-F` at a failure terminal is exactly `-F`, regardless of episode duration. Stalling until the native timeout therefore cannot make failure cheaper. A successful trajectory retains discounted base return `-F + (F + finish) * gamma^(T-1)`, so faster official finishes remain preferable.

Optional instantaneous speed and slip terms exist for experiments. Production keeps both coefficients at zero. Motion terms are suppressed across game-driven teleports and recovery transitions.

### Lap curriculum and evaluation boundary

Training starts each vector instance at one lap and advances that instance from 1 to 2 to 3 laps after one official success at each level. The curriculum changes only the two verified native lap-count words after restoring the same race-start snapshot. Actions, observations, physics, course, rider, and terminal definitions are unchanged. Affine Lock, Bat, and Clifford provide native Puffer precedents for success-driven per-environment curricula.

Every evaluation forces three laps before reset. `target_laps=3` and equality between `three_lap_success_rate` and `success_rate` are acceptance invariants. A short 8,192-decision run that could not advance its training curriculum still reported `target_laps=3` in its in-process post-train evaluation, directly exercising this boundary.

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

The adapter regression harness in [`tests/test_waverace64.cpp`](../../tests/test_waverace64.cpp) covers the ABI shape, action mapping, internal frameskip, body basis, reset contract, guest halfword lane, buoy side, lap-wrap continuity, official misses and disqualification, discount-correct failure shaping, shortened one-lap official finish, full production three-lap finish, B/stick-Y interventions, observation ranges, deterministic baselines, and vector affinity/ownership. Fresh integration runs cover the central CUDA evaluation, post-train reset, and CPU dispatch paths; the unit harness does not invoke those central paths.

### Adapter acceptance and remaining parity gaps

| Gate | Status |
| --- | --- |
| Shortened one-lap official finish fixture | **PASS:** official finish at update 1,070 with action hash `c6ae00920fd86802`. |
| Unmodified production three-lap finish using only the 43 observations | **PASS:** frameskip 4, 2,334 decisions, 9,336 guest updates, score 87,582.3, zero misses, and official success. |
| B and stick-Y authoritative interventions | **PASS:** B changed the authoritative trace in 5 of 6 probes; stick Y changed it in all 6. |
| Generic Puffer regression | **PASS:** stock CartPole CPU evaluation repeated byte-for-byte at fixed seed; stock Breakout CUDA build and evaluation exited zero. |
| Multi-trace interpreter parity, including B and R action regimes | **PENDING** |
| Post-reset secondary RNG and whole-RDRAM parity | **PENDING** |

The retained deterministic rerun exited zero in 4.7 s. Its no-op baseline ended after 2,997 decisions with `perf = 0`. Three fixed-seed random baselines ended after 85, 90, and 85 decisions with `perf = 0.0639`, `0.0556`, and `0.0293`; the harness requires zero official successes for these baselines. These controller fixtures establish task reachability and regression behavior. They do not establish policy learning.

## Performance and learning acceptance

All production measurements below use the repaired upstream 5.0 port, exact production ROM, 43 observations, five action heads, 128 agents, 16 worker threads, one buffer, frameskip 4, horizon 64, minibatch 2,048, asynchronous rollout, and the current runtime archive. The measured host is an NVIDIA GB10 with 10 Cortex-X925 cores, 10 Cortex-A725 cores, and driver 580.95.05. Trainer throughput is exact `agent_steps / uptime`; whole-process resource figures come from GNU time and include startup, checkpoints, and shutdown. CPU equivalents are `(user seconds + system seconds) / process wall`.

Seeds 606, 707, and 808 used deployment commit `10604beca0299ed7909383dfb478c4166788a821` and trainer SHA-256 `8a3a71a782ff02d208cc9a2032e2d0daa3582e3b1c85c0bc422b7e9d3391862c`. Seeds 909 and 1001, plus the fresh evaluation and regression checks, used deployment commit `12a3ec660fa248d48dc3e2fe0c5a9c771c0e1814` and trainer SHA-256 `96a63bad5d70e52daebc9ed8907a0d7ee6e8cb720f99b24be7a498eda3a3e757`. Their patch-equivalent local audit commits are `99b651a23851846de64ab659e84c824fcdbb1dfb` and `9230cec67e3c86f0e93a138c9c64985855428506`; the respective deployment/local pairs have identical Git tree IDs `1769e55c54f93b38580e2346941438ef35a59142` and `a0301fd48c82538ddfeb3d5f794a3c483273099e`. Both cohorts retain the same effective training configuration shown above. The current CPU evaluator SHA-256 is `78910b63eb1ec471c7d4677613b60c27d5cd072d2ab133415d2dbe79cca39817`. Deployment runtime commit `e3f56302898a98ec7f7b20ca35fc1b5de69fe890` and local audit commit `bb824a9a19a77a1cab239d0c783b7052c81dc116` share tree ID `8bfe379221513fd4109e056b875d1a486c30b905`; that tree produced `libwr64.a` SHA-256 `021cc9d7edc4d4ad9848dedbea161c70c98a629c7628e446d9bab260d06a0b5f`.

| Training seed | Decisions | Trainer uptime | Decisions/s | Guest updates/s | Process wall | CPU equivalents | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 606 | 5,242,880 | 169.338 s | 30,961.1 | 123,844.5 | 170.27 s | 16.87 | 1.550 GiB |
| 707 | 5,242,880 | 169.593 s | 30,914.5 | 123,658.1 | 170.53 s | 16.87 | 1.550 GiB |
| 808 | 5,242,880 | 169.492 s | 30,932.8 | 123,731.3 | 170.44 s | 16.87 | 1.551 GiB |
| 909 | 5,242,880 | 169.929 s | 30,853.4 | 123,413.4 | 170.89 s | 16.88 | 1.551 GiB |
| 1001 | 5,242,880 | 170.006 s | 30,839.3 | 123,357.3 | 170.95 s | 16.88 | 1.551 GiB |

The five-seed mean is 30,900.2 decisions/s and 123,600.9 guest updates/s. The population standard deviation across these runs is 46.7 decisions/s, or 0.151% of the mean. A retained topology screen found the best tested frameskip-4 cell at 16 threads and one buffer. Agent counts 128, 256, and 512 differed by at most 0.8% in the retained single-run screen, while 128 agents used 1.55 GiB versus 4.89 GiB at 512. Frameskip 4 delivered 17.9% more guest updates/s than the matched frameskip-1 cell. Two and four buffers reduced throughput under the tested OpenMP placement.

The current architecture is CPU-bound. The production runs consumed about 16.88 CPU equivalents. Across 1,072 device-wide dashboard snapshots from all five runs, GPU utilization had mean 4.17%, median 3%, and range 1% to 8%. These samples are integer NVML snapshots for the whole device, not process-specific continuous telemetry. The earlier topology screen is useful for configuration choice but used horizon 32 and one short run per cell, so it is not substituted for the production figures above.

| Learning result | Acceptance criterion | Status |
| --- | --- | --- |
| Untrained Puffer policy baseline | Three seeds, 807 episodes, zero successes; `perf` ranged from `0.0647` to `0.0705` | **PASS** |
| Strict-PBRS pilot | 5,242,880 decisions, zero successes, held-out `perf=0.0266` | **PASS as a rejected ablation** |
| Production training | Five complete 5,242,880-decision runs with checkpoints and exact outcome logs | **PASS** |
| Fresh stochastic three-lap evaluation | Common held-out seed and at least 512 episodes per checkpoint | **PASS with material seed sensitivity:** 93.28%, 99.02%, 29.50%, 93.00%, and 99.41% |
| Argmax three-lap evaluation | 128 deterministic replicas per checkpoint | **MIXED:** seeds 707, 909, and 1001 finished 128/128; seeds 606 and 808 finished 0/128 |
| Three-lap learning | Nonzero official success on unmodified three-lap episodes | **PASS in all five stochastic policies** |

For common stochastic evaluation seed 7001, checkpoints 606, 707, 808, 909, and 1001 finished 486/521, 507/512, 154/522, 478/514, and 509/512 episodes. Four of five exceeded 92.9%; the pooled rate is 2,134/2,581, or 82.68%. The 29.50% seed-808 result is retained rather than hidden by the pooled average.

Checkpoint 707 remains the retained production checkpoint because it combines strong stochastic evaluation with the best argmax result: 507/512 stochastic at `perf=0.980537`, then 128/128 argmax at `perf=0.991829`, 47 checkpoints, and zero misses. Checkpoint 1001 had the highest common-seed stochastic rate at 509/512 but lower argmax `perf=0.925126`. Checkpoint 808 proves the fixed budget remains seed-sensitive. It finished 154/522 stochastically and 0/128 with argmax.

Each argmax count comprises 128 replicas of one deterministic initial-state and action trajectory. It is deterministic regression and ranking evidence. The stochastic held-out evaluations measure the rollout distribution.

Checkpoint 606 was also evaluated across held-out seeds 7001, 7002, and 7003. It finished 1,469/1,555 official three-lap episodes, or 94.47%. Random and untrained controls had zero successes. A compile, a short rollout, or a shortened one-lap scripted finish is not counted as learning evidence.

### Retained production artifacts

The evidence files below are retained only on the `spark` deployment host and are not vendored into either source checkout. Seeds 606, 707, and 808 are under `/home/spark-advantage/wr64-results/curriculum-screen-10604bec`; seeds 909 and 1001 are under `/home/spark-advantage/wr64-results/curriculum-production-12a3ec66`. Checkpoint and training-log identities are:

| Seed | Final checkpoint SHA-256 | Training log SHA-256 |
| ---: | --- | --- |
| 606 | `f06c36dbba2598555890db1c0e0d349e0175d320fde7e9c247d81b371fa9f301` | `7fb3e2bfbbb54166b7065e36a95908024e35cd27d593df5c568439503bce6975` |
| 707 | `8bbd6ce65587bf8b331e238476b4245fc86903798658ffaa20a385963022c7d4` | `d84638f3824e2560ad0dbc95baff2df6756f623f90deb71ef783fccdf6cc0ef0` |
| 808 | `bd8be566eb08f79bacf58c5f9f61f44073cb784c33e097ea7c5a6946ffbbc91d` | `56e1cf82f044ececef48758f64a0dbdf48ef6f4b77b67ee88b3b613423807a25` |
| 909 | `f6e88691a8e2ac26d8fa782d43739755700c323b7017f0a5ff27c1145a6d2e93` | `5e371a0132ccd0f71d51bb88f8d079d45047557a1bae2bbcea7327538ede83a3` |
| 1001 | `904d4c84575a47d585273c1322c8a9f109fc9bb89c9c42a3d54e7e83d083cc18` | `3f0b77be89b14f1eebbb6417456c70ab8b09ae23f541076ad51080e109c06435` |

The retained seed-707 stochastic and argmax evaluation logs have SHA-256 `9d877f84984a390d169c9b63202cd1b3ddd45ba7826d5295eb5e14fb73640a1c` and `ef3cb5b4f51d3084d065fa5fbb53574992eaf17b4b6f1894095fef82e133f9ab`. All machine-readable evaluation lines report `target_laps=3.000000`, and every retained evaluation log reached its complete `CUDA_EVAL` result line.

## Audit

[`AUDIT.md`](AUDIT.md) traces the prior claims, the evidence behind each verdict, the repairs already made, and the remaining CUDA, performance, and learning gaps.
