# Playback candidate launch and verification

This folder publishes the reviewed launcher and compact CPU evidence for the revision 4 playback candidate. It does not change the immutable release-r1 archive or the existing viewer on port 18771. It contains no human gameplay capture, credentials or closed native trajectory archive.

The app uses one simulation process and one presentation-only process. The simulation advances in fixed 20 ms control steps. The renderer receives 72 finite qpos values, a source tick and reset generation; it never steps dynamics. Rendering runs at at most 20 Hz with one in-flight request and one latest pending snapshot. An old reset generation cannot publish a new image. A renderer failure remains distinct from physics failure. Explicit paused `frame:true` stepping still waits for its identified image. See [app documentation](../../app/README.md) for protocol details.

`launch_viewer.py` verifies the complete app source manifest and every prepared file pin. It compares actual server/worker configuration, environment, controls and executable against the prepared identity and the caller's expected binary SHA-256. It requires a fresh run on loopback port 18772, starts paused, and confirms tick-zero state and a matching PNG hash before writing `STARTED.json`. Startup failure signals only the new process group after checking its PID/start/session identity, including orphaned workers if the new Node leader has exited. It refuses cleanup after a detected PID reuse. Optional `--cpus` applies only to this launcher and its new children; omitting it retains the host's permitted CPU set.

Prepare a fresh run using the revision 4 app and an executable supporting `--render-only`, then launch on Spark:

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

The app passed 39 CPU tests; its 147 manifest entries were hash-verified. The launcher passed seven mock tests, including altered input files, binary mismatch, record-write failure after spawn, group escalation and PID-reuse refusal. `LAUNCHER-R2-RECEIPT.json` pins the reviewed source, prior version, diff and successful output. The first Windows-only test failure is preserved: the fixture initially referenced Linux's `SIGKILL` constant without mocking it. The corrected test supplies that constant without changing Linux launcher semantics. No process-control mock sends a real signal or starts a viewer.

`PUBLICATION.json` records the public copy hashes and the copied-source test run. `APP-R4-RECEIPT.json` records source verification; original machine-local paths in these receipts are provenance, not portable dependencies.
