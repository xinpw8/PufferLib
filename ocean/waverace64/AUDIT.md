# Wave Race 64 implementation audit

## Executive finding

Claude did not invent the static Wave Race 64 core. The repository contains 1,203 native functions produced from a byte-matching US Rev 1 build, a headless runtime, real cartridge DMA, controller injection, overlay dispatch, and an independent interpreter comparison. The repaired runtime now has bit-exact authoritative environment state for one pinned trace through a native failure terminal.

Several claims around that core were materially wrong or unsupported. The ASCII screen was presented like a playable Wave Race display even though it contained no game pixels. Its original position, heading, progress, and gate telemetry came from RAM scans that were later retracted. The display used 60 Hz for a 20 Hz game-update path, labeled B as a brake, and allowed an unresolved indirect call to become a no-op. The first Puffer integration lived in a package whose own `pyproject.toml` said version `4.0.0`, despite its `pufferlib-5.0-lavin` directory name. The simulator has always run on host CPU.

The defensible conclusion is narrower. A real statically recompiled game core existed underneath a misleading demo and an incorrect initial environment. The repairs establish a fixed-course CPU simulator in the native PufferLib 5.0 trainer, a 57-value state observation with local wave samples, native speed, and power, plus a human-readable state evaluator. Production selects per episode from a finite validated pool of complete cartridge-booted race-start states. A controlled renderer-free simulator benchmark measured a 1.44% median throughput reduction for 128 variants, while the paired full-trainer benchmark found no measurable penalty. Two complete randomized-wave runs sustained more than 56,000 decisions/s. Selected seed 903 completed 128/128 deterministic races on validation wave seed 2902 and 128/128 on unseen seed 3902; stochastic evaluation completed 510/512 on each. The simulator remains host-CPU-bound, the two training seeds do not estimate a broader population, and broad game parity remains unproven. Earlier OBS43 and OBS55 checkpoints are incompatible.

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
4. complete fixed-wave OBS57 training and fresh evaluation tied to hashed checkpoints;
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
| The environment is currently proven performant. | The selected randomized-wave run completed 10,485,760 decisions at 56,575.81 decisions/s. Under an identical renderer-free simulator protocol, K=128 randomized and fixed medians were 56,314.635 and 57,135.018 decisions/s. | **Supported.** Complete randomized training, the paired trainer benchmark, and simulator microbenchmarks agree. Simulation remains CPU-bound. |
| A short training smoke test proved the agent could race. | Earlier reward and observation bugs allowed distance gaming, hid the required buoy side, used a short-discount horizon, and sometimes tested shortened episodes. Current evidence uses complete OBS57 randomized-wave runs and fresh official three-lap evaluation. | **False for the original smoke; current learning is supported.** Selected seed 903 finished 128/128 deterministic and 510/512 stochastic episodes on each of two non-training wave pools. |
| B, Z, and R could all be removed as inert inputs. | Nintendo documents real semantics for all three. The deterministic action probe shows A and Z have the same authoritative throttle trajectory, while R changes that trajectory. Targeted mid-race/recovery interventions make B change the authoritative trace in all 6 probes. | **Unsupported.** The final learner removes only redundant Z and retains B and R. |
| The current state evaluator reproduces the N64 graphics. | It draws a custom Raylib scene from an immutable state projection and uses the repository's shared Puffer GLB instead of a rider and jet ski. It does not consume cartridge models, textures, framebuffer, audio, RSP output, or RDP output. Its compact edge HUD mirrors truthful Time Trials information, and its filtered vertical camera anchor exposes wave-relative Puffer motion. | **False as graphical parity.** It is supported as a human-readable visualization of selected simulator state. |

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

The new pure water sampler has a separate, narrower exactness result. The checked-in adapter harness compares `wr64_water_height` bit-for-bit with the recompiled cartridge function `func_8004D30C` at 4,096 deterministic live-field points, covers both interpolation branches, and confirms that the pure query leaves live RDRAM unchanged. A separate implementation characterization compared 262,144 points across 64 randomized complete water fields with zero float-bit mismatches under `-ffp-contract=off`. This validates the translation against the recompiled function. It is not an independent Mupen comparison of the water grid's evolution and does not widen the trace-level game parity claim. An A/B intervention that flattened the live water field diverged from the authoritative baseline by the second guest update, proving that waves affect simulation rather than only presentation.

Direct reset-time calls to `func_8004F790` were rejected after instrumentation showed they do not reproduce the live race-start distribution. The active snapshot follows the final generator call with 57 complete native update-chain calls and contains scripted large waves, nonzero water velocities, rider contact, and related RNG state. Replaying only the low-level generator or water update produced materially narrower height ranges and an incoherent combination of new water with fixed rider state.

Production instead builds `wave_variants` complete cartridge boots through the exact mode-state-2 race-start boundary. A deterministic tested transform of `wave_seed` and variant index, using fixed index salt `0xB9DCF3C0`, changes the initial `osGetTime` value. The salt is an empirically screened deterministic transform, not cryptographic randomness. Seed pairs 42/902 and 902/2902 each produced 128 unique boot-time, full-RDRAM, and complete-water hashes per seed with zero cross-seed overlap. Each donor's complete RDRAM delta is stored in checked 4 KiB pages against the canonical snapshot; native stacks and recompilation contexts remain owner-specific. Construction streams donors, validates route and snapshot scalar invariants, reconstructs every variant byte-for-byte, and rejects exact duplicate complete water fields. Reset restores the owner's snapshot, applies the selected pages, writes the lap target, recomputes the variant-specific vertical origin, and then projects observations. Variant selection is a per-environment odd-stride permutation with exact replay and complete K-step coverage. `puf_eval_reset` rewinds that permutation. Fixed mode preserves the original pinned `WR_BOOT_OS_TIME` snapshot.

The all-128 seed-902 acceptance sweep compared every reconstructed owner with a sequential native donor on complete 8 MiB reset RDRAM, `State`, all 57 observations, transition buffers, render projection, and host-visible machine scalars. It then compared complete RDRAM and host-visible state after every decision in two distinct 32-decision scripts. All comparisons passed. The pool contained 128 unique complete water, height, and velocity fields. A second 128-entry pool using held-out seed 2902 was also internally unique and had zero overlap with seed 902 in boot-time values, full-RDRAM hashes, complete water, height, or velocity fields. Native stack and recompilation-context bytes were not transplanted or claimed donor-identical. A 1,024-reset fixed-action stress test replayed exactly, visited all 128 fields, and reported zero recovery, fault, or terminal events. This establishes coherent authentic reset variants and deterministic reconstruction, not independent interpreter parity for every variant.

The runtime floating-point scope canonicalizes the complete AArch64 FPCR to zero before initialization, reset, projection, step, and render work, then restores the caller's exact token. Hostile-FPCR tests cover the full reset digest, ordinary and terminal steps, state refresh, render capture, and exact host restoration. The first direct `wr_env_step` also enters the scope before its initial `getcontext`, preventing a suspended guest context from saving hostile host controls.

The evaluator capture has also been tested as a read-only state projection. Two captures must have identical logical content and hashes, every captured course pass point must match the adapter geometry, every one of the 1,089 water values must match `wr64_water_height` bit-for-bit, native clocks, speed, and power must match live game fields, live RDRAM must remain unchanged, and capture without rendering must leave `env->client` null. The retained capture hash is `f2b411ab54afcf26` with 59 course nodes.

The separate renderer harness passed on an isolated X display. It kept `Client` null through 192 training decisions and autoresets, verified the Shift+Up rising-edge control toggle, preserved full simulator core state across repeated `puf_render` calls, loaded the shared Puffer GLB as a valid Raylib model, unloaded it cleanly, and produced byte-identical captures with pixel hash `25cf8bb430bd8f98`. Pure captured state retained hash `f2b411ab54afcf26`. Headless and rendered 192-decision trajectories matched with hash `bb75cd7a071f6551`; the 48-frame moving trace ended at guest-update tick 96 with state hash `3217579d69426366`. The filtered vertical camera anchor exposed 12.055 pixels of exact teal-body centroid motion and 21 pixels of silhouette-bottom motion over 4.8 s of simulated time. Real X11 key injection separately produced POLICY, HUMAN, POLICY across two Shift+Up chords. The pixel hash is a regression baseline on the tested software-rendered Raylib/OpenGL stack, not a portable cross-driver promise or an N64 framebuffer reference.

The hardened post-change isolation run initialized 16 machines, executed 512 guest updates per machine, compared serial and parallel hashes, and repeated the parallel run three times. It passed with zero RDRAM, stack, recomp-context, machine-state, or trajectory mismatches. The retained deployment-host log is `/home/spark-advantage/wr64-recomp/runtime/isolation_acceptance_20260823T192430Z_4098593.log`; its SHA-256 is `7c910f98e02b05b2dcc4b022cdcd5b6b3103c419a673e01c1cb4b652d6faa3af`.

## Engineering repairs

| Defect | Repair | Acceptance evidence | Remaining limit |
| --- | --- | --- | --- |
| Guessed player and course fields | Production reads the active decomp-derived physics object, `RiderStruct`, and course graph. Historical candidates moved to `wr_legacy_probe.h`. | Frame-aligned interpreter parity for selected state; strict identity tests. | Other courses and riders unsupported. |
| Hidden buoy passing side | Observation exposes the signed pass point using the course lateral vector and the next pass point. | Pass-point geometry tests and a scripted three-lap controller using the public observation buffer. | Other courses remain outside the task. |
| Dynamic water hidden from the learner | Observation indices 43 through 54 expose a rider-local 4 by 3 stencil at forward offsets `-64,64,192,384` and lateral offsets `-96,0,96`. | Bit-exact observation placement test and exact sampler comparison with `func_8004D30C`. | This is local height only, not complete water or wave state. |
| Native speed and power hidden from the learner | Observation 55 exposes the cartridge physics-speed field and observation 56 exposes the official power level. | ABI placement, range, native speed conversion, and deterministic fixture tests. | These two fields make OBS55 checkpoints incompatible. |
| Snapshot began 800 ms into the live race | Boot now captures mode state 2 with native lap and total clocks both exactly zero. | Reset contract, native clock, live-control, and repeated restore tests. | The fixed snapshot still covers only Sunny Beach Time Trials rider 0. |
| Every reset reused the same wave field | Production selects a coherent race-start state from a finite pool of complete cartridge boots. Checked 4 KiB deltas reconstruct donor RDRAM while each environment keeps its own native stack and context. Fixed mode remains available for parity regressions. | All 128 seed-902 donors matched reconstructed owners across reset state and two 32-decision scripts; complete fields were unique, exact replay passed, the held-out seed-2902 pool had zero overlap, and selected seed 903 passed current policy evaluation. | Pool size is finite and supported only for the scoped Sunny Beach task. |
| No salient PufferLib human evaluation | `puf_render` now draws course topology, buoy sides, the shared Puffer oriented by the captured full body basis, a sampled water surface, minimap, and a compact Wave Race-style Time Trials HUD from an immutable `WR64RenderState`. Native clocks, speed, power, splits, misses, and target laps drive its labels. | Deterministic read-only capture and model-lifecycle regression plus CPU evaluator PNG capture support. | Presentation is custom and has no N64 pixel, cartridge-asset, camera, or audio parity. |
| Camera translation hid wave bob | Horizontal tracking remains direct while the vertical camera anchor follows rider height through a 0.60 s low-pass filter and snaps on reset or recovery. | The moving renderer trace exposed 12.055 pixels of Puffer body-centroid motion and 21 pixels at the silhouette bottom without changing simulator state. | Camera behavior is evaluator presentation, not cartridge-camera parity. |
| Terminal badge over an invisible autoreset race | The evaluator freezes the captured official terminal state while leaving the RL core's same-transition autoreset intact; Enter exposes the already-reset next race. Terminal-aware capture exits before episode two. | Current full-race capture retained 5,372 source frames from time zero through the official three-lap FINISH with `terminal=1`. | This is renderer lifecycle behavior, not a change to simulator terminal semantics. |
| Hold-to-own input caused ambiguous handoff | Shift+Up is a rising-edge HUMAN/POLICY toggle. Policy inference continues during human control and only the submitted action is overridden. | Pure toggle regression plus real X11 POLICY to HUMAN to POLICY sequence. | CPU and CUDA inference trajectories are not asserted equivalent. |
| Evaluation rendering could contaminate training | Renderer allocation and the 33 by 33 display mesh are lazy behind `puf_render`; training never calls it and `Client` remains null. | Renderer harness kept `Client` null through 192 training decisions and showed cadence-independent simulator state. | Simulation and 12 water observation queries still execute on host CPU. |
| Observation ABI changed without checkpoint boundary | `OBS_SIZE` is now 57 and observations 0 through 54 retain their OBS55 meanings. | Compile-time ABI assertion and CPU/CUDA parameter-shape checks. | Every OBS43 and OBS55 checkpoint is incompatible; no migration is defined. |
| Raw absolute height near 39 in the old feature scale | Height is centered on race-start Y and divided by 100. Vertical velocity and the full body basis are exposed. | Deterministic and random observation range checks. | Tested traces do not establish global bounds for every game state. |
| Six action heads with redundant Z | Final ABI is `{15,9,2,2,2}` for stick X, stick Y, A, B, and R. Z is fixed off because it duplicates A. | Nintendo manual, direct A/Z comparison, and B and stick-Y effects in all 6 authoritative intervention probes. | Broader B/R interpreter traces pending. |
| Sparse or cancelled partial-navigation signal | Configured mode credits each new maximum route frontier and verified checkpoint once. Strict terminal-cancelled PBRS remains available as mode 0. | Exact frontier reward regression plus 510/512 stochastic finishes on validation seed 2902 and unseen seed 3902. | The broader training-seed distribution remains unknown. |
| Learner silently clipped rewards to `[-1, 1]` | Wave Race opts out of the generic learner clamp, preserving configured terminal and shaping magnitudes. | Compile-time adapter gate, reward regression, and two complete randomized-wave OBS57 training runs. | A broader seed sweep remains pending. |
| Discount made long failures cheaper than short failures | Every nonterminal transition now charges `reward_fail * (1-gamma)`, making the discounted task return of a zero-miss failure exactly `-reward_fail` at any duration. | Held-A failure, no-op failure, discounted-return regressions, and randomized-wave OBS57 learning. | Broader reward-setting sweeps remain outside the task. |
| Short-horizon `gamma=0.995` with frameskip 1 | The checked-in configuration uses frameskip 2, horizon 128, 10,485,760 decisions, GAE lambda `0.98`, and gamma `0.9997499687421851`; trainer gamma is injected into the environment. | Frameskip and gamma-coupling tests plus two complete randomized-wave runs. This preserves prior guest-update exposure, per-second discount, and a 12.8 s recurrent horizon. | Continuous process-specific GPU telemetry was not retained. |
| Fixed three-lap sparse exploration | Each environment trains on a success-driven 1 to 2 to 3 lap curriculum. Fresh, reused-vector, and CPU evaluation hooks force the official three-lap target. | Direct curriculum regression; one-lap and three-lap scripted fixtures; fresh and post-train target-lap invariants. | Training logs intentionally mix curriculum levels. |
| Unmapped indirect calls became no-ops | `proutSprintf_recomp` is mapped. Unknown targets are fatal. | Runtime isolation acceptance checks both cases. | Interpreter coverage of all dynamic targets is finite. |
| Wrong boot RNG seed | First `osGetTime` returns the measured US Rev 1 seed before the game resets time. | 638-scan authoritative bit parity. | Secondary post-reset time stream remains unmatched. |
| Unsafe guest access and broad address mapping | Alias-safe `memcpy` accessors, correct guest halfword lanes, exact 8 MiB read/write mapping, and inaccessible guards. | Subword probes, fault tests, and retained isolation run. | Generated code still depends on the runtime's guest-memory model. |
| Process-global mutable machine state | Runtime state moved into `WRMachine`; owner is derived from RDRAM, with TLS used only as a cache. | 16-instance serial/parallel full-state equality across three repeats. | Optional profiler is intentionally serialized. |
| Relocatable `Env` objects invalidated saved contexts | Custom vector initialization allocates the exact array once and initializes in place. | Snapshot owner and vector ownership tests. | Environment objects remain immovable by contract. |
| OpenMP initialization could pin rollout creators | Caller affinity is captured and restored before rollout pthread creation. | Affinity equality tests. | A current bound versus unbound benchmark remains unmeasured. |
| Old Puffer 4 integration | Adapter ported to the native upstream PufferLib 5.0 trainer and CPU environment ABI. | Two complete randomized-wave OBS57 CUDA training runs, fresh CUDA and CPU evaluation, deterministic adapter harness, and full-race capture. | The recompiled simulator remains host CPU code. |
| Post-train evaluation inherited partial training episodes or curriculum targets | The trainer now waits for rollout workers, forces three laps, resets every environment, clears transition state, uploads the reset observations, zeros recurrent state, reinitializes action RNG state, and synchronizes before evaluation. | Direct reset coverage plus fresh OBS57 CUDA results with `target_laps=3`. | A retained current reused-vector post-train result is not documented. |
| Misleading live controls and pacing | Runtime telemetry now says `damp waves`, `slide`, `20 updates/s`, and `not game rendering`; absolute sleeps replace the busy spin. | Diagnostic smoke coverage. | This utility is not the human state evaluator and does not measure training. |

## PufferLib 5.0 design review

The repaired adapter now follows the engineering patterns required by the native 5.0 backend:

- a flat `float` observation ABI with compile-time `OBS_SIZE = 57`;
- five multidiscrete action heads with compile-time sizes `{15,9,2,2,2}`;
- direct `puf_init`, `puf_reset`, `puf_step`, `puf_close`, `puf_log`, and state-rendering `puf_render` entry points;
- no per-step heap allocation in headless training;
- lazy evaluator allocation, so training leaves `Client` null and performs no Raylib calls or 33 by 33 render-state capture;
- internal frameskip, so Puffer performs one inference per two guest updates and ten decisions per simulated second;
- exact autoreset semantics that preserve the terminal transition buffers;
- render-only terminal pause, leaving headless training and autoreset unchanged;
- persistent Shift+Up control ownership with policy inference kept synchronized during human play;
- a flat float `Log` structure with episode-count aggregation;
- address-stable custom vector initialization for saved native contexts;
- per-instance machine state and copy-on-write reset backing;
- reset-only authentic boot-state selection from a validated deterministic pool, with no `puf_step` or renderer randomization work;
- restored caller affinity after OpenMP teams;
- clean post-train evaluation initialization for environments, transition state, recurrent state, and action RNGs;
- a Wave Race-only evaluation reset that forces the official three-lap target without changing generic environment lifecycle;
- environment potential discount coupled directly to learner gamma.

The current observation is compact rather than complete. It exposes the reward-potential inputs, target/pass geometry, velocity, recovery, body basis, 12 local water-height samples, native physics speed, and official power. Full RDRAM and the complete water field remain hidden. Strict Markov sufficiency is not claimed. OBS57 adds native speed and power after the unchanged OBS55 fields; loaders require a newly trained 57-input network.

The evaluator consumes an immutable, pointer-free projection containing rider kinematics, body basis, inputs, native clocks, lap splits, speed, power, route and terminal state, course nodes and signed pass points, plus a 33 by 33 local water tile. The renderer turns that projection into a custom low-poly scene and compact Wave Race-style Time Trials HUD. It replaces the rider and jet ski with the repository's 1,199,624-byte `resources/shared/puffer.glb`, SHA-256 `6e5e201b2d08c4eae48f04d9a715ef7b5e6dbb13bffa3c6903ea656730ce7644`, positioned and oriented by the authoritative body basis. It omits Championship rank and opponent portraits because the environment has one rider. Its 0.60 s filtered vertical camera anchor makes wave-relative Puffer motion visible. Model loading is lazy and confined to the evaluator; the renderer-free training path neither allocates `Client` nor opens the GLB. This is consistent with first-party Puffer environment design in separating compact training observations from an evaluation-only visualization, but visual polish itself is not evidence of simulation parity.

The configured reward uses official course topology and official miss/finish flags. Each new maximum route frontier earns credit once, following the dense distance-progress pattern used by Puffer's Drone racer while preventing oscillation from farming reward. Verified checkpoint credit parallels the discrete sector rewards in Whisker Racer. A route-node advance accompanied by a miss is not counted as a successful checkpoint. Game-driven teleports and recovery discontinuities do not create speed or progress reward. The success-driven per-environment lap curriculum follows established native Puffer curriculum patterns in Affine Lock, Bat, and Clifford.

The deterministic adapter harness includes shortened one-lap and unmodified three-lap scripted fixtures through the public 57-value observation buffer. The one-lap fixture finishes at native time 87,835 ms with action hash `64157b7ea07f2a23`. The three-lap fixture finishes in 259,780 ms with splits 79,922/92,115/87,743 ms, zero misses, and action hash `6c5a285ee76ce8e0`. These fixtures run with wave randomization disabled and test fixed-snapshot reachability and regression behavior. Prior fixed-wave policy learning is established separately by complete training and fresh CUDA evaluation; the fixtures do not characterize randomized-wave policy quality or replace throughput measurement.

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

## Randomized reset and learning acceptance, historical fixed-wave evidence, and remaining items

The checked-in randomized reset contract has direct simulator throughput, exhaustive pool reconstruction, held-out-pool separation, reset-stress evidence, two complete trainer runs, a compatible selected checkpoint, fresh CUDA and CPU evaluation, a live human evaluator, and a terminal-aware full-race MP4. The training pool used wave seed 902. Held-out seed 2902 had zero overlap with it in boot ticks, full RDRAM, complete water, height, or velocity fields. Additional seed 3902 was first evaluated after policy selection.

Under the identical renderer-free 128-environment, 16-thread adapter rollout protocol, fixed trials measured 55,380.376, 57,135.018, and 58,949.920 decisions/s. K=128 randomized trials measured 56,314.635, 28,184.067, and 57,529.228 decisions/s. The medians are 57,135.018 and 56,314.635 decisions/s, a 1.44% randomized reduction. The low randomized sample is retained as observed system noise. K=128 initialization measured 8.647 s and 106,968 KiB maximum RSS at N=1, and about 3.4 s and 1,893,084 KiB at N=128. This isolates steady adapter rollout and pool lifetime; it is not a complete CUDA trainer result.

Two renderer-free PufferLib 5.0 runs each completed 10,485,760 decisions from random initialization. Seed 902 took 186.54 s wall time, or 56,211.86 decisions/s. Selected seed 903 took 185.34 s, or 56,575.81 decisions/s and 113,151.62 guest updates/s, with 1,683% CPU and 1,688,236 KiB maximum RSS. Its final training batch reported 100% three-lap success and no failure terminal. An identical 2,097,152-decision paired trainer benchmark took 43.81 s fixed-wave and 43.02 s randomized-wave, so reset-time pool selection had no measurable trainer penalty. The simulator remained host-CPU code; CUDA handled network inference, recurrent state, tensors, and PPO learning.

Exact retained artifacts are under `/home/spark-advantage/wr64-results/obs57-authentic-waves/audit`. The all-128 acceptance source and log SHA-256 values are `fbac23774ddd8f7d01bb80367f27828d03e2cbabb51df50d0a3ac58983533586` and `8d9d3295ef4bf9627aa533142ef825d36cce6a5d8770b098b059a762fdf0e694`. The 1,024-reset stress source and log values are `41617a89dea347118dbb1906c372bf9927067c015b6113cf80bf77978a6c16d6` and `db7792dc4b144d9482a163b637ffb08231b1180b46e9eebb1c1f58e5925265e5`. The throughput source and log values are `d05569e25c83d85e4dd166ca2a25ac684ba92a97d6f3654c334f7ee5cc784392` and `6c6a370ab5b21294bd0ca3e9162f232f7f63394efb5ecdf526d8a0bd753c36c7`. The tick-schedule source and log values are `a5cf0194c678df14bf788d938c8180829d091d295bb9f40447bc05b2c2330b52` and `3d7b4d971e8195767b23fbd1f9f7f593469203afe43883826edbccb82bed8497`. Final runtime ABI/environment, adapter, native-trainer build, renderer, and renderer-UBSan log hashes are `0073b21b657007a8db38bc4b9faef269b3af2d969eb58492f5a83d94dd856618`, `9ce38dde8259e3834941ea59f954e2aa0737a7a7b536c81f9a9750bcee9ae52e`, `2f8bd52527094f9b0037c1eed2934a020e17f96f1579378f8c2c35caa5994e3c`, `49fce4c37b35771c297e04d736833b0b3f94a3fda2312453cb8416f2fe7a4f9e`, and `49fce4c37b35771c297e04d736833b0b3f94a3fda2312453cb8416f2fe7a4f9e`. `SHA256SUMS-final`, itself SHA-256 `b81d6a03a193550109f38e85e5620c69f4df693cc04691d721591b844add8961`, covers every retained source, log, and renderer image in the directory.

The selected 438,272-byte, 109,568-parameter checkpoint is `/home/spark-advantage/wr64-results/obs57-authentic-waves/checkpoints/waverace64/obs57-authwave-s903-final/0000000010485760.bin`, SHA-256 `91217a7eb1ff5f4d553b678c206c5acaa727461bcf45cfd4f0266f7c2e0f62bf`. Fresh deterministic CUDA processes completed 128/128 official three-lap episodes on seed 2902 and 128/128 on unseen seed 3902. A separate one-environment seed-2902 run completed an exact odd-stride permutation of all 128 variants at 128/128. Stochastic CUDA completed 510/512 on each pool. Each stochastic run had one generic failure and one disqualification; all runs had zero safety timeout and zero environment fault. A separate CPU process loaded exactly 109,568 floats and completed one seed-3902 race in 896 decisions with zero miss or adverse terminal.

The current full-race MP4 is `/home/spark-advantage/wr64-results/obs57-authentic-waves/video/waverace64-puffer-s903-unseen3902-full-race.mp4`, 21,331,534 bytes, SHA-256 `847e092fdd1328a020a0f64e9260a20463615d3141838835ad575d7ba59034d0`. It is H.264 High/yuv420p at 960 by 540 and 60 frames/s. It contains 5,372 time-zero-through-terminal source frames plus a 240-frame frozen finish, 5,612 decoded frames total, and lasts 93.53 s. Full decode and independent frame inspection passed; the official finish is 89.519 s with no episode-two contamination.

The following seed-900 and seed-901 results are retained historical fixed-wave provenance.

The clean fixed-wave OBS57 frameskip-2 seed-900 run executed 10,485,760 decisions in 201.845 s of trainer uptime. This is 51,949.565260 policy decisions/s and 103,899.130521 native updates/s. The renderer remained inactive. The binary used for this retained measurement has SHA-256 `f20a62342714f2c0698d356435849d0f3069fbed703e3e7c83ba510c15cff9d2`; it predates the evaluator-only Puffer model change and authentic per-episode wave-pool selection.

The selected seed-901 OBS57 checkpoint is 438,272 bytes. It is retained at `/home/spark-advantage/wr64-results/obs57-seeds/checkpoints/waverace64/obs57-fs2-s901/0000000010485760.bin` with SHA-256 `eaf2d9be637f5d03a95bb1d6ff9c40096867e977dfcdbd1eb8f94df855f277b5`.

Fresh deterministic CUDA evaluation with action seed 1901 completed 128/128 official three-lap finishes. It reported zero misses, generic failures, disqualifications, safety timeouts, or environment faults, `target_laps=3`, native finish time 82,607 ms, lap splits 29,643/26,029/26,935 ms, 1,654 guest updates, and mean speed 1,018.898 game units/s.

Held-out stochastic CUDA evaluation with action seed 3901 completed 509/512 official three-lap episodes (`0.994141`). It reported 0.330078 mean misses, two generic failures, one disqualification, no timeout or fault, and `target_laps=3`. Among successes, native finish time averaged 84,200.930 ms and lap splits averaged 28,945.953/27,441.498/27,813.461 ms. Mean episode length was 1,683.535 guest updates and mean speed was 1,040.705 game units/s.

A fresh standalone CPU evaluator built from the same committed OBS57 source loaded all 109,568 checkpoint floats and completed one deterministic official three-lap episode in 784 policy decisions. It passed all 48 checkpoints with zero misses, failures, disqualifications, safety timeouts, or environment faults. The binary SHA-256 is `62f8a3be97fda9659f204e4a6e08aac09b57aaa3ded2e19c31b89fc2116c73ab`. The retained log is `/home/spark-advantage/wr64-results/obs57-seeds/evals/final-cpu-puffer-det-s1901.log`, SHA-256 `b31035bc62f6386a72ca2a28c8ad26c034d633be57473bceedcee55bcdbe5b44`. This smoke proves checkpoint compatibility and official completion on the CPU evaluator; it does not assert bit-identical CPU and CUDA inference.

These results establish fixed-wave OBS57 learning for one selected training seed under deterministic and held-out stochastic action selection. They do not establish randomized-wave learning or the broader training-seed distribution.

The prior OBS55 evidence remains historical. Seeds 707 and 708 each executed 5,242,880 decisions at 30,777.828655 and 30,713.28 decisions/s by trainer uptime. Both used 1,687% CPU with maximum RSS 1,625,884 and 1,625,936 KiB. Their trainer binaries had SHA-256 `893c63d0cde3fb9d8073232fe5b1c98a543f53e00732c01b5488e7c7817952f` and `e96263824d6083e08f8d18d57aa415dcf68c2ca2a9260d1b8192cbf90a59633c`. Their 437,248-byte checkpoints had SHA-256 `a6696e3888ca472712071aa9fd6b82b377e3ddf956db41ca2082488c3145fc59` and `83382a3f31141a1645a5be2cb8c31696480066838faaee70c8527138722027dd`; both are incompatible with the current network. Historical seed 707 finished 128/128 deterministic and 386/514 stochastic episodes; historical seed 708 finished 0/128 deterministic and 410/515 stochastic episodes. Those results demonstrate the superseded OBS55 contract, not current OBS57 performance.

The earlier OBS55 learned-policy capture, its 108.433 s full-race MP4, and the pre-Puffer OBS57 rider-and-jet-ski recording remain historical artifacts. The fixed-wave OBS57 CUDA-policy Puffer MP4 is retained at `/home/spark-advantage/wr64-results/obs57-seeds/video/waverace64-obs57-s901-puffer-full-race-20260823.mp4`. It is H.264 High/yuv420p at 960 by 540 and 60 frames/s, contains 5,348 decoded frames, lasts 89.133 s, is 23,783,474 bytes, and has SHA-256 `42e3db2597955bbc43d5078735635afa590c9dcc38fc316cf481640d12a12b60`. It starts at native time zero, contains the complete deterministic 82.607 s official race, and retains more than 6 s of the frozen finish. Frames at 0, 10, 82, and 88 s were inspected independently for start, live Puffer wave motion, final lap, terminal state, and absence of a captured X11 cursor.

| Item | Status and required evidence |
| --- | --- |
| Randomized-wave OBS57 PufferLib 5.0 performance | **PASS.** Selected full trainer run sustained 56,575.81 decisions/s and 113,151.62 guest updates/s. Paired fixed/random trainer jobs were 43.81/43.02 s. |
| Randomized-wave OBS57 learning | **PASS.** Selected seed 903 completed 128/128 deterministic races on held-out seed 2902 and 128/128 on unseen seed 3902; stochastic was 510/512 on each. |
| Native race timing | **PASS.** Selected video finished in native time 89,519 ms; evaluator clocks and lap splits remain cartridge fields. |
| Broader seed sensitivity | **PARTIAL.** Two full training seeds learned; selected seed 903 was stronger than seed 902. This is not a population estimate. |
| Compatible randomized-wave checkpoint | **PASS.** Selected 438,272-byte seed-903 OBS57 checkpoint has the documented hash and exact 109,568-float CPU load. |
| Human state evaluation | **PASS.** Valid shared Puffer model, compact truthful Time Trials HUD, native clocks/speed/power, two-way human toggle, terminal freeze, visible wave-relative motion, and live display-98 deployment are verified. |
| Current OBS57 Puffer full-race MP4 | **PASS.** Selected randomized-wave policy from time zero through the frozen official finish; 93.53 s, 5,612 decoded frames, full decode clean. |
| Continuous process-specific GPU and per-core telemetry | **PENDING.** Renderer and capture runs must be excluded from training measurements. |
| Broad controller parity | **PENDING.** Add interpreter traces that exercise steering, stick Y, B, R, wave interaction, recovery, misses, and a successful finish. |
| Secondary RNG and whole-RDRAM parity | **PENDING.** Implement a measured post-reset time model before making either claim. |
| CUDA simulator | **UNIMPLEMENTED.** Requires a separate device-native core and fresh parity program. |

The detailed environment contract, commands, observations, actions, reward, terminals, logs, and empirical results are in [`README.md`](README.md).
