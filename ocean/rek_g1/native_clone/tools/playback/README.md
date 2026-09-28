# Playback candidate launch and verification

This folder contains the unchanged reviewed launcher and its historical CPU evidence, now used with app revision 7. The current app manifest is `6be9fa74673eca108da8c07cd414d27d55b6cc9f625db1ab70e2f5c3e6f02cca`; all 168 listed payload files plus the manifest were copied byte-for-byte from the frozen candidate. The immutable release-r1 archive and earlier validation receipts remain unchanged. This folder contains no human gameplay capture or credentials.

The app uses one simulation process and one presentation-only process. The simulation advances in fixed 20 ms control steps. The renderer receives 72 finite qpos values, a source tick and reset generation; it never steps dynamics. Rendering runs at at most 20 Hz with one in-flight request and one latest pending snapshot. An old reset generation cannot publish a new image. A renderer failure remains distinct from physics failure. Explicit paused `frame:true` stepping still waits for its identified image. See [app documentation](../../app/README.md) for protocol details.

Revision 7 schedules absolute monotonic deadlines, with one step in flight, an early-wake check and explicit reporting when wall-clock debt above 100 ms is rebased. Pause/resume resets the schedule. Every input and state remains synchronously recorded; replaceable health-file writes are limited to 250 ms during ordinary steps, with immediate lifecycle/error updates. The pacing metric continues to use actual simulation steps and elapsed wall time. Final performance and deployment conclusions belong to their separately closed runtime receipts.

The current [worker template](../worker-template.json) selects one arena and the existing batch-2 models. Preparation requires encoder SHA256 `0e1cf37a7c1bafe870741b8a3de2560ae7a2b0cfe14c5ac8cef4d8a34d78207f` and decoder SHA256 `20b49c9df1a54dc3a211d0d86c2ebe3ccc1de67883a984a7b227af76af7aacb3`, retains the full GPU/Bot1 environment guards, and records arena count and required controller batch in the prepared identity. The former four-arena path remains supported. A one-arena control-step rate must not be multiplied by the old four-arena count.

`launch_viewer.py` verifies the complete app source manifest and every prepared file pin. It compares actual server/worker configuration, environment, controls and executable against the prepared identity and the caller's expected binary SHA-256. It requires a fresh run on loopback port 18772, starts paused, and confirms tick-zero state and a matching PNG hash before writing `STARTED.json`. Startup failure signals only the new process group after checking its PID/start/session identity, including orphaned workers if the new Node leader has exited. It refuses cleanup after a detected PID reuse. Optional `--cpus` applies only to this launcher and its new children; omitting it retains the host's permitted CPU set.

Prepare a fresh run using revision 7 and the verified executable supporting `--render-only`, then launch on Spark:

```sh
python3 tools/playback/launch_viewer.py --app /absolute/path/to/app \
  --run /absolute/path/to/fresh-prepared-run \
  --manifest-sha256 VERIFIED_APP_MANIFEST_SHA256 \
  --binary-sha256 VERIFIED_NATIVE_BINARY_SHA256
```

This is source publication, not a deployment receipt. The frozen release executable lacks the new renderer protocol. No 1x speed or official simulation parity claim follows from these tests. Actual performance and deployment require separately closed runtime evidence.

Validation:

```sh
python3 -B -m unittest discover -s tools/playback -p test_launch_viewer.py -v
node --test app/league/*.test.cjs
```

Revision 7 passed [52 CPU tests](../../app/validation/tests-r7.txt). Its genuine-model preparation test is skipped locally when the large model files are absent; the unchanged preparation source has a [separate actual Spark pass](../../app/validation/tests-r6-real-models.txt) using both real batch-2 and batch-8 models. That test confirms one/four-arena acceptance and rejects wrong/swapped models, unsupported counts and CPU mode. The launcher remains SHA256 `e8ea7a2ba243d8a548ee99f7b35229679b17803936a1812d31f50efb48b146e0` and retains its seven passing mock tests, including altered input files, binary mismatch, record-write failure after spawn, group escalation and PID-reuse refusal. `LAUNCHER-R2-RECEIPT.json` pins that source, prior version, diff and output. The first Windows-only test failure remains preserved; its fixture required a mock for Linux's `SIGKILL` constant. No process-control mock sends a real signal or starts a viewer.

`PUBLICATION.json` and `APP-R4-RECEIPT.json` describe the earlier revision 4 publication, including its original README and 39-test result. They remain historical receipts. `APP-R7-COPY.json` records the current app copy and the three worker-template field changes. Original machine-local paths in these receipts are provenance, not portable dependencies. No identical tests were rerun for the repository copy.
