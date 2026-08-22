# Wave Race 64 implementation audit

## Executive finding

Claude did not invent the static Wave Race 64 core. The repository contains 1,203 native functions produced from a byte-matching US Rev 1 build, a headless runtime, real cartridge DMA, controller injection, overlay dispatch, and an independent interpreter comparison. The repaired runtime now has bit-exact authoritative environment state for one pinned trace through a native failure terminal.

Several claims around that core were materially wrong or unsupported. The ASCII screen was presented like a playable Wave Race display even though it contained no game pixels. Its original position, heading, progress, and gate telemetry came from RAM scans that were later retracted. The display used 60 Hz for a 20 Hz game-update path, labeled B as a brake, and allowed an unresolved indirect call to become a no-op. The first Puffer integration lived in a package whose own `pyproject.toml` said version `4.0.0`, despite its `pufferlib-5.0-lavin` directory name. The simulator has always run on host CPU.

The defensible conclusion is now measured. A real statically recompiled game core existed underneath a misleading demo and an incorrect initial environment. The repairs establish a fixed-scenario CPU simulator in the native PufferLib 5.0 trainer, stable production throughput near 30.9 K policy decisions/s, and learned official three-lap finishes. The simulator remains host-CPU-bound, seed sensitivity is material, and broad game parity remains unproven.

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
4. current implementation source and version metadata;
5. historical RAM scans, terminal plots, and short training runs.

The [official Nintendo controller page](https://www.nintendo.com/eu/media/downloads/games_8/emanuals/nintendo_8/Manual_Nintendo64_WaveRace64_EN.pdf#page=5) is authoritative for player-facing input semantics. It defines A as throttle, B as wave damping, R as sliding, and the stick as handling and center-of-gravity control. Z duplicates A throttle.

## Claim-by-claim verdict

| Claim or implication | Evidence | Verdict |
| --- | --- | --- |
| The process executes real Wave Race 64 code. | `libwr64.a` contains 1,203 generated native game functions. The runtime DMAs the pinned cartridge, boots the real menu state machine, executes overlays, and reaches official race terminals. | **Supported.** |
| The ASCII box is Wave Race 64 rendering. | RSP, RDP, framebuffer rendering, and display output are stubbed. The terminal draws a character and trail from sampled state. `puf_render` is empty. | **False.** It is a telemetry plot. The current tool says so explicitly. |
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
| The environment was already suitable for high-throughput learning. | The original transcript reported about 20.9 K steps/s and compared it with an unrelated CUDA proxy. The repaired production system sustains a five-seed mean of 30.90 K policy decisions/s and 123.60 K guest updates/s. | **False when originally claimed; measured after repair.** Simulation remains CPU-bound. |
| A short training smoke test proved the agent could race. | Earlier reward and observation bugs allowed distance gaming, hid the required buoy side, used a short-discount horizon, and sometimes tested shortened episodes. Current evidence uses complete 5,242,880-decision runs and fresh official three-lap evaluation. | **False for the original smoke; supported by the new production runs.** |
| B, Z, and R could all be removed as inert inputs. | Nintendo documents real semantics for all three. The deterministic action probe shows A and Z have the same authoritative throttle trajectory, while R changes that trajectory. Targeted mid-race/recovery interventions make B change the authoritative trace in 5 of 6 probes. | **Unsupported.** The final learner removes only redundant Z and retains B and R. |

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

The repaired `play` utility now reads authoritative physics and rider structures, labels itself `WR64 HEADLESS TELEMETRY`, identifies the ASCII state plot as non-rendering, paces with an absolute 20 Hz sleep, uses alias-safe guest reads, and labels B/R according to the official controls. Its fixed low-CPU display loop does not measure training throughput.

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

The retained hardened 16-instance isolation log is `runtime/isolation_acceptance_20260822T050325Z_3003381.log`; its SHA-256 is `805b8384d97306f77c043734727a5cd0ffbece35088574bbb4ebf9ba165d852f`.

## Engineering repairs

| Defect | Repair | Acceptance evidence | Remaining limit |
| --- | --- | --- | --- |
| Guessed player and course fields | Production reads the active decomp-derived physics object, `RiderStruct`, and course graph. Historical candidates moved to `wr_legacy_probe.h`. | Frame-aligned interpreter parity for selected state; strict identity tests. | Other courses and riders unsupported. |
| Hidden buoy passing side | Observation exposes the signed pass point using the course lateral vector and the next pass point. | Pass-point geometry tests, a zero-miss scripted three-lap controller, and learned official three-lap finishes using only public observations. | Other courses remain outside the task. |
| Raw absolute height near 39 in the old feature scale | Height is centered on race-start Y and divided by 100. Vertical velocity and the full body basis are exposed. | Deterministic and random observation range checks. | Tested traces do not establish global bounds for every game state. |
| Six action heads with redundant Z | Final ABI is `{15,9,2,2,2}` for stick X, stick Y, A, B, and R. Z is fixed off because it duplicates A. | Nintendo manual, direct A/Z comparison, B effects in 5 of 6 authoritative intervention probes, and stick-Y effects in 6 of 6. | Broader B/R interpreter traces pending. |
| Sparse or cancelled partial-navigation signal | Production mode credits each new maximum route frontier and verified checkpoint once. Strict terminal-cancelled PBRS remains available as mode 0. | Exact frontier reward regression plus production learning screen. | Mode 2 intentionally lets valid partial progress survive a later failure. |
| Learner silently clipped rewards to `[-1, 1]` | Wave Race opts out of the generic learner clamp, preserving configured terminal and shaping magnitudes. | Compile-time adapter gate plus native train/eval acceptance. | None within the fixed task. |
| Discount made long failures cheaper than short failures | Every nonterminal transition now charges `reward_fail * (1-gamma)`, making the discounted task return of a zero-miss failure exactly `-reward_fail` at any duration. | Held-A failure, 2,997-decision no-op failure, three-lap discounted-return regressions, and production learning. | None within the fixed task. |
| Short-horizon `gamma=0.995` with frameskip 1 | Production uses frameskip 4, horizon 64, GAE lambda `0.98`, and gamma `0.9995`; trainer gamma is injected into the environment. | Frameskip and gamma-coupling tests, temporal pilots, and full production runs. | None within the fixed task. |
| Fixed three-lap sparse exploration | Each environment trains on a success-driven 1 to 2 to 3 lap curriculum. Fresh, reused-vector, and CPU evaluation hooks force the official three-lap target. | Direct curriculum regression; one-lap and three-lap scripted fixtures; fresh and post-train target-lap invariants. | Training logs intentionally mix curriculum levels. |
| Unmapped indirect calls became no-ops | `proutSprintf_recomp` is mapped. Unknown targets are fatal. | Runtime isolation acceptance checks both cases. | Interpreter coverage of all dynamic targets is finite. |
| Wrong boot RNG seed | First `osGetTime` returns the measured US Rev 1 seed before the game resets time. | 638-scan authoritative bit parity. | Secondary post-reset time stream remains unmatched. |
| Unsafe guest access and broad address mapping | Alias-safe `memcpy` accessors, correct guest halfword lanes, exact 8 MiB read/write mapping, and inaccessible guards. | Subword probes, fault tests, and retained isolation run. | Generated code still depends on the runtime's guest-memory model. |
| Process-global mutable machine state | Runtime state moved into `WRMachine`; owner is derived from RDRAM, with TLS used only as a cache. | 16-instance serial/parallel full-state equality across three repeats. | Optional profiler is intentionally serialized. |
| Relocatable `Env` objects invalidated saved contexts | Custom vector initialization allocates the exact array once and initializes in place. | Snapshot owner and vector ownership tests. | Environment objects remain immovable by contract. |
| OpenMP initialization could pin rollout creators | Caller affinity is captured and restored before rollout pthread creation. | Affinity equality tests and retained fast/slow CPU-class screen. | A repeated bound versus unbound production matrix remains unmeasured. |
| Old Puffer 4 integration | Adapter ported to the native upstream PufferLib 5.0 trainer and CPU environment ABI. | Native CUDA-policy and CPU-evaluator builds, deterministic adapter harness, five production runs, and generic CartPole/Breakout regressions. | None within the fixed task. |
| Post-train evaluation inherited partial training episodes or curriculum targets | The trainer now waits for rollout workers, forces three laps, resets every environment, clears transition state, uploads reset observations, zeros recurrent state, reinitializes action RNG state, and synchronizes before evaluation. | Fresh CUDA, reused-vector post-train, and standalone CPU evaluations all reported `target_laps=3`; stock CartPole CPU repeated byte-for-byte and Breakout CUDA evaluated successfully. | None within the fixed task. |
| Misleading live controls and pacing | Runtime telemetry now says `damp waves`, `slide`, `20 updates/s`, and `not game rendering`; absolute sleeps replace the busy spin. | Six-second smoke held 20 updates/s and consumed about 0.08 s of host CPU time. | This utility is diagnostic only. |

## PufferLib 5.0 design review

The repaired adapter now follows the engineering patterns required by the native 5.0 backend:

- a flat `float` observation ABI with compile-time `OBS_SIZE = 43`;
- five multidiscrete action heads with compile-time sizes `{15,9,2,2,2}`;
- direct `puf_init`, `puf_reset`, `puf_step`, `puf_close`, `puf_log`, and no-op `puf_render` entry points;
- no per-step heap allocation;
- internal frameskip, so Puffer performs one inference per four guest updates;
- exact autoreset semantics that preserve the terminal transition buffers;
- a flat float `Log` structure with episode-count aggregation;
- address-stable custom vector initialization for saved native contexts;
- per-instance machine state and copy-on-write reset backing;
- restored caller affinity after OpenMP teams;
- clean post-train evaluation initialization for environments, transition state, recurrent state, and action RNGs;
- a Wave Race-only evaluation reset that forces the official three-lap target without changing generic environment lifecycle;
- environment potential discount coupled directly to learner gamma.

The current observation is compact rather than complete. It exposes the reward-potential inputs, target/pass geometry, velocity, recovery, and body basis. Full RDRAM and water state remain hidden. Strict Markov sufficiency is not claimed.

The production reward uses official course topology and official miss/finish flags. Each new maximum route frontier earns credit once, following the dense distance-progress pattern used by Puffer's Drone racer while preventing oscillation from farming reward. Verified checkpoint credit parallels the discrete sector rewards in Whisker Racer. A route-node advance accompanied by a miss is not counted as a successful checkpoint. Game-driven teleports and recovery discontinuities do not create speed or progress reward. The success-driven per-environment lap curriculum follows established native Puffer curriculum patterns in Affine Lock, Bat, and Clifford.

The retained deterministic adapter rerun exited zero in 4.7 s. The shortened one-lap fixture finished at update 1,070 with action hash `c6ae00920fd86802`. A controller reading only the 43 production observations completed the unmodified three-lap race at frameskip 4 in 2,334 policy decisions and 9,336 guest updates, with score 87,582.3, zero misses, and official success. This proves task reachability through the public adapter. Independent policy training and fresh-process evaluation now provide separate learning evidence below.

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

## Unresolved acceptance items

| Item | Status and required evidence |
| --- | --- |
| Current PufferLib 5.0 performance | **MEASURED.** Five production runs average 30,900.2 decisions/s and 123,600.9 guest updates/s at about 16.87 CPU equivalents and 1.55 GiB RSS. Three-run device-wide GPU sampling averaged 4.20%. |
| Production learning | **PASS.** Complete 5,242,880-decision runs learned official three-lap finishes; the best retained checkpoint finished 507/512 stochastic and 128/128 argmax fresh-process episodes. |
| Seed sensitivity | **MEASURED AND MATERIAL.** Common-seed stochastic success for five training seeds was 93.28%, 99.02%, 29.50%, 93.00%, and 99.41%. The fixed budget produced nonzero learning in all five but did not guarantee a strong checkpoint. |
| Continuous process-specific GPU and per-core telemetry | **PARTIAL.** Existing NVML samples are device-wide dashboard snapshots. Continuous process attribution and power/clock traces were not retained. |
| Broad controller parity | **PENDING.** Add interpreter traces that exercise steering, stick Y, B, R, wave interaction, recovery, misses, and a successful finish. |
| Secondary RNG and whole-RDRAM parity | **PENDING.** Implement a measured post-reset time model before making either claim. |
| CUDA simulator | **UNIMPLEMENTED.** Requires a separate device-native core and fresh parity program. |

The detailed environment contract, commands, observations, actions, reward, terminals, logs, and benchmark placeholders are in [`README.md`](README.md).
