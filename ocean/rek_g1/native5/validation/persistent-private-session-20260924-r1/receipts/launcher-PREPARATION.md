# Persistent client fault-tracing launcher

Prepared only. Do not execute until the coordinating agent approves the maintenance restart and installs the separately reviewed bridge build.

Parent: `/home/spark-advantage/rek-training/timing500-ppo-refresh-live-20260924-r1/root-campaign/relaunch.sh`, SHA256 `ca556b4688ee2d14ea2e0e3baf0d0c23f20d3f79a1375943ff145c572f96793b`. The parent file is preserved locally as `relaunch.baseline.sh`.

Runtime environment changes: add `BOX64_SHOWSEGV=1` and `BOX64_SHOWBT=1`. `BOX64_LOG=1`, STRONGMEM=2, WEAKBARRIER=0, Wine, Box64, game, evidence recorder, renderer, process/prefix scope checks, and all other runtime settings remain unchanged. The bridge pin is updated from `11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d` to the separately built SkipIntro/native-Home variant `5a2edac6c586f1ea401d92e0086ebfc468dbc591e2bb11115aef056682280e7a`. Reviewed build artifact: `C:/rekagent/work/spark-startup-skip-20260924-r1/artifacts/bridge/RekUiBridgeAgent.dll`; source: sibling `variant/windows/RekUiBridgeAgent`. The old authorship wording is removed in this copy only.

Planned launch, exclusively by the coordinating agent after review:

```sh
bash /home/spark-advantage/rek-training/persistent-private-session-20260924-r1/relaunch.sh persistent-diagnostic-r1
```

The launcher only starts the isolated client. Arena entry and policy execution remain separate native commands. It contains no kill, retry, per-round recycle, or policy-input operation. Its existing startup guard refuses an occupied REK process or dedicated prefix. The inherited disabled pair-capture configuration copy is unchanged; this is not an authorization to alter any other config.

Private fault logs, created only upon launch:

`/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-persistent-diagnostic-r1/stderr.txt`

Sibling `stdout.txt`, `unity.log`, `game.pid`, `game.initial-stat.txt`, and eventual exit receipts use the unchanged launcher convention. The output directory is mode 0700, with umask 077. Raw registers/stacks remain private, outside Git and public reports.

Keep this instrumented process across actual supported private-AI fights. An unsupported pairing can end a policy attempt but must not force client recycling or terminate fault logging. The earlier one-shot failed before a supported fight and did not establish crash-capture efficacy. r97's subsequently reported survival beyond 13 minutes with policy off disproves an unconditional 155-second process timer; active-control workload remains an unproven lead.

SHOWSEGV makes original Wine-handled fault reporting visible at the unchanged LOG=1 level; SHOWBT requests native/emulated stacks before forwarding the signal to Wine. Installed Box64 source: `src/libtools/signals.c:782,1177,1251,1313`; emitted signals: `src/os/emit_signals_linux.c:47`. PE/JIT unwind may truncate and software-only exceptions may bypass the Linux signal handler. Handled-exception logging can add latency or alter reproduction. This is diagnostic instrumentation, not a demonstrated crash fix or scoring-parity claim.

Verification is syntax, exact-diff, and file-hash/readback only. No launcher execution, process stop, bridge connection, GPU test, or live-game interaction was performed in preparing this copy.
