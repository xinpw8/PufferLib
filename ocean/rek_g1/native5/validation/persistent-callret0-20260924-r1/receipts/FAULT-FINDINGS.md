# CALLRET=0 outcome, private diagnostic summary

The initial client exited 5 at 2026-09-24T12:56:43Z. CALLRET=0 did not prevent a fatal failure. This failure is an execution-at-null signal, different from the previous instrumented bad-pointer read instructions. The corruption origin remains unknown.

Runtime directory: `/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-persistent-callret0-r1`. Private `stderr.txt`: 2286320 bytes, SHA256 `a590070a38f0b945d8b8dc7e6169da5d9cc09acede934a65149e733a486c838a`.

At line 23035 Box64 reports an attempt to run at NULL. Line 23036 emits signal 11 with guest instruction pointer and fault address both null. There is no instruction to decode or method range to assign at address zero. The later CoreCLR fail-fast address is not the original target.

Using the installed Box64 executable's symbols, the five Box64 native frames resolve, from the signal reporter outward, to `EmitSignal`, `CheckExec`, `Run`, `EmuRun`, `pthread_routine`; two libc frames follow. The emulated backtrace contains only the unresolved null target. No previous emulated caller is recovered.

Installed source `/home/spark-advantage/codexrook-runtime/src/box64` explains the immediate path: `src/emu/x64run.c:63` logs the zero instruction pointer; `:73` calls CheckExec. `src/os/emit_signals_linux.c:113` checks execute/read protection and `:119` emits the execution-access signal. `src/libtools/threads.c:306` is the thread wrapper, which runs EmuRun via DynaRun. Seeing that wrapper does not prove the thread initially began at a null function. The prior control transfer that selected zero remains unknown.

Saved `/proc` receipts pin host PID3110517, namespace PID56250 and start ticks94209601 to the exact REK target and initial stat. Actual flags were CALLRET=0, STRONGMEM=2, WEAKBARRIER=0, SHOWSEGV=1, SHOWBT=1 and LOG=1. No runtime settings or processes were changed by this monitoring subtask.

The first completed controlled round ended 11:11 at 12:55:15.345Z: 1071 source states, 9.0263 Hz, p95 source interval128.656 ms, Unity/wall0.999015. The later `callret-s1802-retry2` terminated early: final source12:56:39.1018599Z, native round3, remaining85.6994 s, score5:3; 318 source states at9.3546 Hz, p95 interval123.9687 ms. Source states advanced one Unity frame each. Earlier instrumented baseline control was approximately24–26 Hz. These changed cadences confound efficacy comparisons.

Resource context at12:56:09.220Z: 20logicalCPUs, load1.88/1.87/2.12, MemAvailable102311172 KiB, GPU utilization1%, CPU/memory/I/O PSI avg10 all0. GPU processes were the pre-existing Python process, REK and current policy worker. These snapshots show no sampled saturation; they do not identify the cause of either slowdown or crash. No other process was changed.

Data-quality note: the first `host-resources-20260924T125524Z.json` used schema v1 and incorrectly tokenized process names containing spaces. Its process CPU/RSS fields must not be used. The corrected schema-v2 `host-resources-20260924T125609Z.json` retains multiword names and supplies the values above. Its memory/GPU fields and the separate game process-identity receipts do not use that faulty tokenization.

Next prepared diagnostic restores the original instrumented baseline and adds only JIT/Loader EventPipe collection. This is a return to baseline plus metadata, not evidence of a fix and not a single-variable comparison against this failed CALLRET0 run.
