# Final staged native verification drivers

These sources are staged only. No native test, GPU process or viewer was launched during preparation.

The unchanged build recipe recompiles combat and the scheduler and preserves the original physics objects. Root supplies isolated `source-r6` and `build-r6` through `REK_CLONE_SOURCE_DIR` and `REK_CLONE_BUILD_DIR`.

```sh
REK_CLONE_SOURCE_DIR=/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/source-r6 REK_CLONE_BUILD_DIR=/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/build-r6 bash /home/spark-advantage/rek-training/rek-native-clone-20260927-r1/tools-r5/build-native.sh
python3 /home/spark-advantage/rek-training/rek-native-clone-20260927-r1/tools-r5/smoke_native.py /home/spark-advantage/rek-training/rek-native-clone-20260927-r1
python3 /home/spark-advantage/rek-training/rek-native-clone-20260927-r1/tools-r5/match_smoke.py
```

The smoke defaults to run-r4 and fresh verification-r5. It preserves all requests/replies and invalid stdout. It reads phases instead of inferring acceptance from elapsed ticks: one inactive attack edge must be rejected with clone telemetry reason 4, never accepted or delivered later; a new active request must receive native acceptance or a recovery/punching rejection. Neutral single-step waits are bounded to 2000 ticks. Invalid-command, reset, numeric, side/mode and renderer checks remain.

The separate match smoke uses fresh match-verification-r2, retains a complete native match and tests that its result stays latched. The short-round smoke's existing throughput measurement is a diagnostic request duration over its tested phases, not a training or real-time manual-play benchmark.

After parent verification, launch_remote.py starts only prepared run-r4 using app-r3, initially paused on the unused loopback port. native_cli.py is the parent's current pinned direct protocol entry point. Both are launch-capable tools and were not executed in this preparation.

CPU contract tests: `python3 -m unittest -v test_smoke_contract.py`. They test a valid/rejected inactive edge, duplicate/deferred edge failure, dynamic phase progression and the bounded wait. Existing full native tests still require the parent's GPU run.
