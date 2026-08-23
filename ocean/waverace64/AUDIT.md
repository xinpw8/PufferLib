# Wave Race 64 implementation audit

## Executive finding

Claude did not invent the static Wave Race 64 core. The repository contains 1,203 native functions produced from a byte-matching US Rev 1 build, a headless runtime, real cartridge DMA, controller injection, overlay dispatch, and an independent interpreter comparison. The repaired runtime now has bit-exact authoritative environment state for one pinned trace through a native failure terminal.

Several claims around that core were materially wrong or unsupported. The ASCII screen was presented like a playable Wave Race display even though it contained no game pixels. Its original position, heading, progress, and gate telemetry came from RAM scans that were later retracted. The display used 60 Hz for a 20 Hz game-update path, labeled B as a brake, and allowed an unresolved indirect call to become a no-op. The first Puffer integration lived in a package whose own `pyproject.toml` said version `4.0.0`, despite its `pufferlib-5.0-lavin` directory name. The simulator has always run on host CPU.

The defensible conclusion is narrower. A real statically recompiled game core existed underneath a misleading demo and an incorrect initial environment. The repairs establish a fixed-scenario CPU simulator in the native PufferLib 5.0 trainer, a 55-value state observation with local wave samples, and a human-readable state evaluator. Two complete current-contract runs measured throughput, and both checkpoints learned official three-lap behavior under stochastic evaluation. Deterministic per-head argmax is sharply seed-sensitive: seed 707 finished 128/128 while seed 708 finished 0/128. The simulator remains host-CPU-bound, broader seed statistics are pending, and broad game parity remains unproven. Earlier 43-input checkpoints remain incompatible.

## Original assignment

The human assignment required:

1. a native Wave Race 64 core capable of running inside PufferLib;
2. actual-game parity rather than a loose proxy;
3. high training throughput;
4. integration with the current PufferLib 5.0 native stack;
5. a CUDA simulator, if feasible, so simulation would not consume host CPU.

The later stop-hook feedback explicitly rejected partial completion of the first three items. A core that runs and trains at low or unmeasured throughput does not satisfy the high-throughput goal.

## Evidence hierarchy

This audit gives evidence weight in the following order:

1. decompilation-derived structures and constants for the exact ROM revision;
2. frame-aligned output from an independent Mupen64Plus pure interpreter;
3. deterministic adapter and multi-instance isolation tests;
4. complete current-contract training and fresh evaluation tied to hashed checkpoints;
5. current implementation source and version metadata;
6. historical RAM scans, terminal plots, and short training runs.

The [official Nintendo controller page](https://www.nintendo.com/eu/media/downloads/games_8/emanuals/nintendo_8/Manual_Nintendo64_WaveRace64_EN.pdf#page=5) is authoritative for player-facing input semantics. It defines A as throttle, B as wave damping, R as sliding, and the stick as handling and center-of-gravity control. Z duplicates A throttle.

## Claim-by-claim verdict

| Claim or implication | Evidence | Verdict |
| --- | --- | --- |
| The process executes real Wave Race 64 code. | `libwr64.a` contains 1,203 generated native game functions. The runtime DMAs the pinned cartridge, boots the real menu state machine, executes overlays, and reaches official race terminals. | **Supported.** |
| The ASCII box is Wave Race 64 rendering. | RSP, RDP, framebuffer rendering, and display output are stubbed. The terminal draws a character and trail from sampled state. | **False.** It is a telemetry plot. The current tool says so explicitly. A separate Raylib state evaluator now implements `puf_render`. |
| The original ASCII position and heading were authoritative game telemetry. | The demo read scan-derived candidates at `0x801C3780` and `0x80227D70`. The decomp-derived player physics object is at `0x80192690`, with position at `+0x44` and forward X/Z at `+0x1434/+0x1438`. | **Unsupported and misleading.** The old addresses are now quarantined in `wr_legacy_probe.h`. |
| The five displayed gate positions and `0x801C3154` were course progress. | Their increments clustered in a burst and did not correlate reliably with route passage. The original project notes later retracted the interpretation. | **False.** Production now follows `RiderStruct+0xC` and the decomp-derived course graph. |
| The screen's speed, position, and heading proved visual or emulator parity. | A text plot cannot prove video output. The original values came from untrusted fields and an incorrect update multiplier. | **False.** Current parity evidence comes from the independent interpreter trace. |
| `60 fps` described the environment step rate. | Interpreter CP0 timing shows one game/controller update every three approximately 60 Hz VI callbacks. `wr_env_step(..., 1)` is a 20 Hz guest update. | **False.** The runtime and adapter now use the verified 20 Hz cadence. |
| S or Down was a brake. | S injected the N64 B button. Nintendo defines B as wave damping, not general braking. | **False.** The live runtime label is fixed to `damp waves`; R is labeled `slide`. |
| `0x800CA27C -> no-op` was an acceptable unresolved indirect target. | The address is `proutSprintf_recomp`, an identified callback. Treating unknown control flow as harmless can silently delete game behavior. | **Unsupported and unsafe.** The callback is mapped and any unknown indirect target now aborts. |
| The old integration was PufferLib 5.0. | The old branch's `pyproject.toml` declares `version = "4.0.0"` and uses the older Python-era architecture. Its directory name does not change the package version. | **False.** The repaired adapter targets upstream branch `5.0` at `ba238f8c` and compiles into `src/pufferl.cu`. |
| PufferLib 5.0 means this environment runs on CUDA. | The policy, rollout tensors, and learner use CUDA. `waverace64.h` is a CPU backend, and `libwr64.a` runs in host vector workers. | **False for simulation.** GPU learning and CPU simulation coexist. |
| Training can avoid touching the CPU. | Every environment step resumes a host `ucontext_t`, executes native C, reads and writes host-mapped RDRAM, and uses POSIX runtime shims. | **False in the current architecture.** |
| The static core had broad parity with the actual game. | One held-A Sunny Beach trace now matches selected authoritative fields bit-for-bit through failure. Secondary RNG, graphics, audio, whole RDRAM, other inputs, success, courses, and modes are outside that proof. | **Partially supported after repair.** Any blanket parity claim remains unsupported. |
| The environment is currently proven performant. | Two complete renderer-free 5,242,880-decision runs measured 30,777.828655 and 30,713.28 decisions/s by trainer uptime. Both used 1,687% CPU and about 1.626 million KiB maximum RSS. | **Measured for two full current-contract runs.** The retained 43-input baseline is not treated as a fully pinned duration-matched comparison, and simulation remains CPU-bound. |
| A short training smoke test proved the agent could race. | Earlier reward and observation bugs allowed distance gaming, hid the required buoy side, used a short-discount horizon, and sometimes tested shortened episodes. Current evidence instead uses two complete 55-input runs and fresh official three-lap evaluation. | **False for the original smoke; stochastic learning is supported in both current seeds.** Deterministic argmax sensitivity and the broader seed distribution remain material limits. |
| B, Z, and R could all be removed as inert inputs. | Nintendo documents real semantics for all three. The deterministic action probe shows A and Z have the same authoritative throttle trajectory, while R changes that trajectory. Targeted mid-race/recovery interventions make B change the authoritative trace in 5 of 6 probes. | **Unsupported.** The final learner removes only redundant Z and retains B and R. |
| The current state evaluator reproduces the N64 graphics. | It draws custom Raylib primitives from an immutable state projection. It does not consume cartridge models, textures, framebuffer, audio, RSP output, or RDP output. | **False as graphical parity.** It is supported as a human-readable visualization of selected simulator state. |

## What the pasted terminal actually showed

The pasted process was attached to a real native recompile. Several lines on its display had different evidentiary value:

| Display element | Audit interpretation |
| --- | --- |
| `WAVE RACE 64` | Reasonable process identity. The executable links the static game core. |
| `recompiled from your cartridge` | Substantially correct for the pinned ROM and static game code. Hardware paths remain headless shims. |
| `Nintendo's own physics` | Supported for the authoritative physics fields on the one parity trace. Too broad as a statement about all traces or effects. |
| Empty ASCII box and `^` | Host-generated top-down state plot. No Nintendo rendering occurred. |
| Position and heading values | Untrusted in the original output because they came from scan-derived candidates. |
| Lap and game state | Read from game structures, although place is not useful in a one-rider Time Trial. |
| `60 fps` and speed | Incorrect time base. The guest gameplay update is 20 Hz. |
| `unmapped indirect target ... -> no-op` | A real runtime defect, now repaired. |

The repaired `play` utility now reads authoritative physics and rider structures, labels itself `WR64 HEADLESS TELEMETRY`, identifies the ASCII state plot as non-rendering, paces with an absolute 20 Hz sleep, uses alias-safe guest reads, and labels B/R according to the official controls. It remains a diagnostic utility. It is distinct from the PufferLib Raylib state evaluator and does not measure training throughput.

## Parity evidence and boundary

The independent oracle pins:

- Wave Race 64 US Rev 1, 8,388,608 bytes, SHA-256 `f35d2423ebcb86eaf86fa935b613c7532b123a7bc50fb74996984c3b02fc3999`;
- Mupen64Plus pure interpreter commit `6dca4c15370ac3e2171ce7b31426695f8f39b460`;
- RSP HLE commit `8a7a472a7172eb2c8725b305eae26818ed7b51a2`;
- real menu selection of Time Trials, rider confirmation, Sunny Beach, two neutral updates, then a fixed held-A trace.

Two 2,400-VI interpreter repeats were internally identical. Two static repeats were internally identical. After dropping one extra pre-game interpreter controller scan, 638 scans align from guest frame 0 through 637.

Both engines produce the same route sequence:

| Guest frame | Lap | Target node | Misses | Game state |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 0 | 0 | 0 | `0x28` |
| 248 | 1 | 2 | 0 | `0x28` |
| 325 | 1 | 4 | 0 | `0x28` |
| 368 | 1 | 9 | 1 | `0x28` |
| 637 | 1 | 9 | 1 | `0x29` |

Primary RNG, active physics position/heading/speed bits, rider route state, official outcome flags, recovery, mode identity, and controller inputs are bit-exact across all aligned scans. The runtime originally returned zero from the cartridge's first `osGetTime`, selecting a different primary physics RNG stream. Recovering the interpreter value `0x56D84BE4` by inverting the common cartridge LCG and using it for the first boot-time call repaired the discrepancy.

The proof remains trace-specific. The secondary `Math_Rand` state at `0x80226F00` differs at guest frame 2 because the headless runtime does not model post-reset time advancement. No listed authoritative field changed through frame 637. Graphics, audio, every writable byte, successful three-lap completion, and alternate action regimes remain unproven.

The new pure water sampler has a separate, narrower exactness result. The checked-in adapter harness compares `wr64_water_height` bit-for-bit with the recompiled cartridge function `func_8004D30C` at 4,096 deterministic live-field points, covers both interpolation branches, and confirms that the pure query leaves live RDRAM unchanged. A separate implementation characterization compared 262,144 points across 64 randomized complete water fields with zero float-bit mismatches under `-ffp-contract=off`. This validates the translation against the recompiled function. It is not an independent Mupen comparison of the water grid's evolution and does not widen the trace-level game parity claim.

The evaluator capture has also been tested as a read-only state projection. Two captures must have identical logical content and hashes, every captured course pass point must match the adapter geometry, every one of the 1,089 water values must match `wr64_water_height` bit-for-bit, live RDRAM must remain unchanged, and capture without rendering must leave `env->client` null. The retained capture hash is `7d5f6487fbfc38ba` with 59 course nodes.

The separate renderer harness was repeated independently with exact results. It kept `Client` null through 192 training decisions and autoresets, verified the Shift+Up rising-edge control toggle, preserved full simulator core state across eight repeated `puf_render` calls, produced two byte-identical compact-HUD 960 by 540 captures with pixel hash `1d5323433008b7ff`, kept headless and variably rendered 96-decision trajectories identical with hash `ffa76503834a3bff`, and produced moving guest-update tick-192 state hash `3d79ad5e60a0b87f`. Real X11 key injection separately produced POLICY, HUMAN, POLICY across two Shift+Up chords. The pixel hash is a regression baseline on the tested software-rendered Raylib/OpenGL stack, not a portable cross-driver promise or an N64 framebuffer reference.

The hardened 16-instance isolation log is retained only on the deployment host at `/home/spark-advantage/wr64-recomp/runtime/isolation_acceptance_20260822T050325Z_3003381.log`; its SHA-256 is `805b8384d97306f77c043734727a5cd0ffbece35088574bbb4ebf9ba165d852f`.

## Engineering repairs

| Defect | Repair | Acceptance evidence | Remaining limit |
| --- | --- | --- | --- |
| Guessed player and course fields | Production reads the active decomp-derived physics object, `RiderStruct`, and course graph. Historical candidates moved to `wr_legacy_probe.h`. | Frame-aligned interpreter parity for selected state; strict identity tests. | Other courses and riders unsupported. |
| Hidden buoy passing side | Observation exposes the signed pass point using the course lateral vector and the next pass point. | Pass-point geometry tests and a scripted three-lap controller using the public observation buffer. | Other courses remain outside the task. |
| Dynamic water hidden from the learner | Observation indices 43 through 54 expose a rider-local 4 by 3 stencil at forward offsets `-64,64,192,384` and lateral offsets `-96,0,96`. | Bit-exact observation placement test and exact sampler comparison with `func_8004D30C`. | This is local height only, not complete water or wave state. |
| No salient PufferLib human evaluation | `puf_render` now draws course topology, buoy sides, rider pose, a sampled water surface, minimap, and HUD from an immutable `WR64RenderState`. | Deterministic read-only capture test plus CPU evaluator PNG capture support. | Presentation is custom and has no N64 pixel, asset, camera, or audio parity. |
| Terminal badge over an invisible autoreset race | The evaluator freezes the captured official terminal state while leaving the RL core's same-transition autoreset intact; Enter exposes the already-reset next race. Terminal-aware capture exits before episode two. | Final source frame shows the official three-lap FINISH; exact capture reported 620 frames and `terminal=1`. | This is renderer lifecycle behavior, not a change to simulator terminal semantics. |
| Hold-to-own input caused ambiguous handoff | Shift+Up is a rising-edge HUMAN/POLICY toggle. Policy inference continues during human control and only the submitted action is overridden. | Pure toggle regression plus real X11 POLICY to HUMAN to POLICY sequence. | CPU and CUDA inference trajectories are not asserted equivalent. |
| Evaluation rendering could contaminate training | Renderer allocation and the 33 by 33 display mesh are lazy behind `puf_render`; training never calls it and `Client` remains null. | Renderer harness kept `Client` null through 192 training decisions and showed cadence-independent simulator state. | Simulation and 12 water observation queries still execute on host CPU. |
| Observation ABI changed without checkpoint boundary | `OBS_SIZE` is now 55 and observations 0 through 42 retain their meanings. | Compile-time ABI assertion and CPU/CUDA parameter-shape checks. | Every 43-input checkpoint is incompatible; no migration is defined and retraining is required. |
| Raw absolute height near 39 in the old feature scale | Height is centered on race-start Y and divided by 100. Vertical velocity and the full body basis are exposed. | Deterministic and random observation range checks. | Tested traces do not establish global bounds for every game state. |
| Six action heads with redundant Z | Final ABI is `{15,9,2,2,2}` for stick X, stick Y, A, B, and R. Z is fixed off because it duplicates A. | Nintendo manual, direct A/Z comparison, B effects in 5 of 6 authoritative intervention probes, and stick-Y effects in 6 of 6. | Broader B/R interpreter traces pending. |
| Sparse or cancelled partial-navigation signal | Configured mode credits each new maximum route frontier and verified checkpoint once. Strict terminal-cancelled PBRS remains available as mode 0. | Exact frontier reward regression plus stochastic official three-lap learning in seeds 707 and 708. | The broader seed distribution remains unknown. |
| Learner silently clipped rewards to `[-1, 1]` | Wave Race opts out of the generic learner clamp, preserving configured terminal and shaping magnitudes. | Compile-time adapter gate, reward regression, and two complete current-contract training runs. | A broader seed sweep remains pending. |
| Discount made long failures cheaper than short failures | Every nonterminal transition now charges `reward_fail * (1-gamma)`, making the discounted task return of a zero-miss failure exactly `-reward_fail` at any duration. | Held-A failure, no-op failure, discounted-return regressions, and stochastic learning in both current seeds. | The broader seed distribution remains unknown. |
| Short-horizon `gamma=0.995` with frameskip 1 | The checked-in configuration uses frameskip 4, horizon 64, GAE lambda `0.98`, and gamma `0.9995`; trainer gamma is injected into the environment. | Frameskip and gamma-coupling tests plus two complete 5,242,880-decision runs. | A broader seed distribution and fully pinned duration-matched old/new comparison remain pending. |
| Fixed three-lap sparse exploration | Each environment trains on a success-driven 1 to 2 to 3 lap curriculum. Fresh, reused-vector, and CPU evaluation hooks force the official three-lap target. | Direct curriculum regression; one-lap and three-lap scripted fixtures; fresh and post-train target-lap invariants. | Training logs intentionally mix curriculum levels. |
| Unmapped indirect calls became no-ops | `proutSprintf_recomp` is mapped. Unknown targets are fatal. | Runtime isolation acceptance checks both cases. | Interpreter coverage of all dynamic targets is finite. |
| Wrong boot RNG seed | First `osGetTime` returns the measured US Rev 1 seed before the game resets time. | 638-scan authoritative bit parity. | Secondary post-reset time stream remains unmatched. |
| Unsafe guest access and broad address mapping | Alias-safe `memcpy` accessors, correct guest halfword lanes, exact 8 MiB read/write mapping, and inaccessible guards. | Subword probes, fault tests, and retained isolation run. | Generated code still depends on the runtime's guest-memory model. |
| Process-global mutable machine state | Runtime state moved into `WRMachine`; owner is derived from RDRAM, with TLS used only as a cache. | 16-instance serial/parallel full-state equality across three repeats. | Optional profiler is intentionally serialized. |
| Relocatable `Env` objects invalidated saved contexts | Custom vector initialization allocates the exact array once and initializes in place. | Snapshot owner and vector ownership tests. | Environment objects remain immovable by contract. |
| OpenMP initialization could pin rollout creators | Caller affinity is captured and restored before rollout pthread creation. | Affinity equality tests. | A current bound versus unbound benchmark remains unmeasured. |
| Old Puffer 4 integration | Adapter ported to the native upstream PufferLib 5.0 trainer and CPU environment ABI. | Two complete current 55-input CUDA training runs, fresh CUDA evaluation, fresh CPU evaluation, and deterministic adapter harness. | A broader seed sweep remains pending. |
| Post-train evaluation inherited partial training episodes or curriculum targets | The trainer now waits for rollout workers, forces three laps, resets every environment, clears transition state, uploads reset observations, zeros recurrent state, reinitializes action RNG state, and synchronizes before evaluation. | Direct reset coverage plus fresh current-contract CUDA and CPU results with `target_laps=3`. | A separate retained current-contract reused-vector post-train result is not documented. |
| Misleading live controls and pacing | Runtime telemetry now says `damp waves`, `slide`, `20 updates/s`, and `not game rendering`; absolute sleeps replace the busy spin. | Diagnostic smoke coverage. | This utility is not the human state evaluator and does not measure training. |

## PufferLib 5.0 design review

The repaired adapter now follows the engineering patterns required by the native 5.0 backend:

- a flat `float` observation ABI with compile-time `OBS_SIZE = 55`;
- five multidiscrete action heads with compile-time sizes `{15,9,2,2,2}`;
- direct `puf_init`, `puf_reset`, `puf_step`, `puf_close`, `puf_log`, and state-rendering `puf_render` entry points;
- no per-step heap allocation in headless training;
- lazy evaluator allocation, so training leaves `Client` null and performs no Raylib calls or 33 by 33 render-state capture;
- internal frameskip, so Puffer performs one inference per four guest updates;
- exact autoreset semantics that preserve the terminal transition buffers;
- render-only terminal pause, leaving headless training and autoreset unchanged;
- persistent Shift+Up control ownership with policy inference kept synchronized during human play;
- a flat float `Log` structure with episode-count aggregation;
- address-stable custom vector initialization for saved native contexts;
- per-instance machine state and copy-on-write reset backing;
- restored caller affinity after OpenMP teams;
- clean post-train evaluation initialization for environments, transition state, recurrent state, and action RNGs;
- a Wave Race-only evaluation reset that forces the official three-lap target without changing generic environment lifecycle;
- environment potential discount coupled directly to learner gamma.

The current observation is compact rather than complete. It exposes the reward-potential inputs, target/pass geometry, velocity, recovery, body basis, and 12 local water-height samples. Full RDRAM and the complete water field remain hidden. Strict Markov sufficiency is not claimed. The old 43-input checkpoint ABI ended when those water features were appended; loaders require a newly trained 55-input network.

The evaluator consumes an immutable, pointer-free projection containing rider kinematics, body basis, inputs, route and terminal state, course nodes and signed pass points, plus a 33 by 33 local water tile. The renderer turns that projection into a custom low-poly scene and HUD. This is consistent with first-party Puffer environment design in separating compact training observations from an evaluation-only visualization, but visual polish itself is not evidence of simulation parity.

The configured reward uses official course topology and official miss/finish flags. Each new maximum route frontier earns credit once, following the dense distance-progress pattern used by Puffer's Drone racer while preventing oscillation from farming reward. Verified checkpoint credit parallels the discrete sector rewards in Whisker Racer. A route-node advance accompanied by a miss is not counted as a successful checkpoint. Game-driven teleports and recovery discontinuities do not create speed or progress reward. The success-driven per-environment lap curriculum follows established native Puffer curriculum patterns in Affine Lock, Bat, and Clifford.

The deterministic adapter harness includes shortened one-lap and unmodified three-lap scripted fixtures through the public 55-value observation buffer. These fixtures test reachability and regression behavior. Current stochastic policy learning is established separately for seeds 707 and 708 by complete training and fresh evaluation; the scripted fixtures do not characterize seed sensitivity or replace throughput measurement.

## Why the current core cannot become CPU-free by changing the binding

The static recompile is structurally a host program. A census limited to the 19 `RecompiledFuncs/funcs_*.c` files found:

- 1,203 `RECOMP_FUNC` definitions;
- 561,734 lines and 20,162,167 bytes of generated C;
- 71,895 `MEM_*` occurrences;
- 73 `LOOKUP_FUNC` call sites.

Each machine also depends on host-only mechanisms:

- `ucontext_t` suspension and a native game stack;
- POSIX mappings, guarded address space, and `memfd` copy-on-write reset state;
- host message-queue, DMA, overlay, fault, and controller shims;
- indirect dispatch through overlay-aware host function tables;
- a memory-resident MIPS register context and 8 MiB guest RDRAM image.

CUDA device code cannot resume a host `ucontext_t`, call these POSIX services, or use the host mappings. Marking the current functions `__device__` would not solve control-flow divergence, guest pointer translation, dynamic overlays, reset state, or runtime services.

A device-native implementation is technically a separate project. It would need to translate or reimplement the gameplay subset, define a GPU memory model, replace OS/runtime behavior, handle dynamic call targets, preserve VR4300 floating-point behavior, and build a new interpreter-parity suite. The earlier 290 M steps/s CUDA proxy measured a simplified surrogate, not this game core, so it provides no evidence for parity or achievable full-game throughput.

CPU-free simulation is therefore infeasible within the present static-recomp architecture. A separate CUDA reimplementation may be research-worthy. It does not follow automatically from the current core, and its parity would begin unproven.

## Current-contract empirical acceptance and remaining items

The complete renderer-free seed-707 and seed-708 runs each executed 5,242,880 decisions with distinct binary artifacts. Seed 707 ran `./puffer-wr64-obs55`, SHA-256 `893c63d0cde3fb9d8073232fe5b1c98a543f53e00732c01b5488e7c7817952f`; seed 708 ran `./puffer-wr64-final`, SHA-256 `e96263824d6083e08f8d18d57aa415dcf68c2ca2a9260d1b8192cbf90a59633c`. Later changes affected renderer-state hashing, the render harness, and documentation rather than the training path, but the binaries remain separate provenance records. Seed 707 used 170.346 s of trainer uptime and 171.28 s of wall time, yielding 30,777.828655 decisions/s internally, 123,111.314618 guest updates/s, and 30,609.995329 wall decisions/s. Seed 708 used 170.704 s of trainer uptime and 171.65 s of wall time, yielding 30,713.28 decisions/s internally, 122,853.13 guest updates/s, and 30,544.01 wall decisions/s. Both used 1,687% CPU; maximum RSS was 1,625,884 and 1,625,936 KiB. Seed 708's time-weighted internal GPU mean was 4.287%; a comparable full-run seed-707 value is not retained. The historical 43-input baseline is not treated as a fully pinned duration-matched comparison.

Both resulting checkpoints are 437,248 bytes. Seed 707 is retained at `/home/spark-advantage/wr64-results/state-eval-obs55-e0a8e2d4/checkpoints/waverace64/state-eval-obs55-s707/0000000005242880.bin` with SHA-256 `a6696e3888ca472712071aa9fd6b82b377e3ddf956db41ca2082488c3145fc59`. Seed 708 is retained at `/home/spark-advantage/wr64-results/state-eval-obs55-seed708-final/checkpoints/waverace64/state-eval-obs55-s708/0000000005242880.bin` with SHA-256 `83382a3f31141a1645a5be2cb8c31696480066838faaee70c8527138722027dd`.

For seed 707, fresh deterministic CUDA evaluation completed 128/128 official three-lap finishes with `perf=0.937920`, score 81,820.898438, 46 checkpoints, zero misses, and zero generic failures, disqualifications, timeouts, or faults. Stochastic action seed 7001 completed 386/514 (`0.750973`) with `perf=0.877817`, score 76,568.945312, 39.085602 mean checkpoints, 2.708171 mean misses, 80 generic failures (`0.155642`), 48 disqualifications (`0.093385`), and no timeout or fault.

For seed 708, fresh deterministic CUDA evaluation completed 0/128 with `perf=0.720770`, score 62,625.097656, 32 checkpoints, one miss, and 128 generic failures; disqualification, timeout, and fault counts were zero. Stochastic action seed 7002 completed 410/515 (`0.796117`) with `perf=0.938844`, score 81,918.937500, 41.429127 mean checkpoints, 3.330097 mean misses, 10 generic failures (`0.019417`), 95 disqualifications (`0.184466`), and no timeout or fault. Every CUDA result reported `target_laps=3`.

Seed 708 therefore learned strong stochastic race behavior. Its failure is specific to deterministic per-head argmax, whose selected modal trajectory completed 0/128, versus 128/128 for seed 707. This demonstrates mode-policy sensitivity across the two training seeds. It does not establish the broader seed distribution.

Fresh deterministic CPU evaluation completed its single episode successfully in 619 decisions with `perf=0.889577`, score 77,603.546875, 39 checkpoints, one miss, and no failure, disqualification, timeout, or fault. This is a functional one-episode check, not a CPU success-rate estimate or CUDA/CPU trajectory-equivalence result.

The earlier learned-policy visual capture retained 150 distinct consecutive 960 by 540 PNGs spanning 30 simulated seconds. Frame 149 showed target gate 51, 13 gates cleared, and zero misses. The terminal-aware CPU capture retained 620 640 by 480 source frames from reset through the official three-lap finish, then stopped before episode two. The primary native CUDA-policy H.264 High/yuv420p recording is 640 by 360 at 60 frames/s, contains 6,506 decoded frames, lasts 108.433 s, is 11,085,404 bytes, and has SHA-256 `041bba61d8f86a4a00e4aec555a9b71a7d47e287696aaba7cc60823f31dbfc1b`. It begins at the first rendered policy transition after a synchronized restart and ends with 120 consecutive frames of the frozen official finish. The final state reports lap 3/3, 46 cleared gates, zero misses, and finish time 106.60 s. This verifies inspectable full-race CUDA policy behavior. It does not widen the trace-specific simulator parity proof or create N64 graphical parity.

| Item | Status and required evidence |
| --- | --- |
| Current 55-input PufferLib 5.0 performance | **MEASURED FOR TWO COMPLETE RUNS.** A broader seed distribution, fully pinned old/new duration-matched comparison, and complete GPU telemetry remain pending. |
| Current 55-input stochastic learning | **PASS IN BOTH SEEDS.** Seed 707 finished 386/514 and seed 708 finished 410/515 official three-lap episodes. |
| Deterministic argmax sensitivity | **MEASURED AND SHARP.** Seed 707 finished 128/128; seed 708 finished 0/128. This is argmax collapse, not absence of stochastic learning in seed 708. |
| Broader seed sensitivity | **PENDING.** Two training seeds establish the contrast but do not estimate the wider distribution. |
| Compatible retained checkpoints | **RETAINED.** Both are 437,248 bytes with distinct hashes documented above. |
| CPU policy execution | **PASS AS A ONE-EPISODE CHECK.** Broader CPU evaluation and action-by-action backend comparison remain pending. |
| Human learned-policy evaluation | **PASS FOR STABLE SEED 707.** Compact HUD, two-way human toggle, terminal freeze, and a full-race MP4 are verified; visual parity with the N64 is not claimed. |
| Continuous process-specific GPU and per-core telemetry | **PENDING.** Renderer and capture runs must be excluded from training measurements. |
| Broad controller parity | **PENDING.** Add interpreter traces that exercise steering, stick Y, B, R, wave interaction, recovery, misses, and a successful finish. |
| Secondary RNG and whole-RDRAM parity | **PENDING.** Implement a measured post-reset time model before making either claim. |
| CUDA simulator | **UNIMPLEMENTED.** Requires a separate device-native core and fresh parity program. |

The detailed environment contract, commands, observations, actions, reward, terminals, logs, and empirical results are in [`README.md`](README.md).
