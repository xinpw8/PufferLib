# Native REK clone and measured evidence

This package preserves the working C++/CUDA clone tested on DGX Spark. It includes the exact source overlay, the browser app and original exported meshes, saved G1 controls, build and test tools, passive transport/backup helpers, and the compiled release with its verification records.

**Full official-game parity remains unverified.** Recovered controller, match and referee functions have targeted tests; authoritative physics, initial state, global callback order, server assets and end-to-end trajectories still require matched official comparisons. This package does not establish superhuman play or a successful learned policy.

## Contents

- `source/`: current source overlay, including the playback candidate's separate renderer protocol and cached reporting. The original 50-file build-r6 source closure remains in the frozen release archive. Relative includes are preserved. The shared legacy evaluator remains separate because its reduced backend does not implement the new direct-command API.
- `dependency_source/`: frozen historical source files associated with the inventoried build dependencies, with original paths and hashes. These preserve the source record; the current build still consumes the pinned Spark objects and libraries.
- `app/`: revision 7 paused-by-default viewer with absolute 20 ms scheduling, continuous commands, original mesh rendering, saved controls, native round/match metrics and detailed recording. Physics and presentation run in separate processes; source manifests preserve earlier revisions.
- `tools/`: pinned build recipe, worker/environment templates, GPU smoke/match test drivers and headless JSON-line launcher. [Playback launch tools](tools/playback/README.md) add fresh-port identity checks and bounded cleanup, with CPU test receipts.
- `passive_support/`: loopback SSH tunnel and append-verified NAS mirror. Explicit `--until-stop` removes the old four-hour expiry; omitting it retains the timed default.
- `validation/`: closed test results, original-function comparisons, dependency inventories, connection repair evidence and the preceding project checkpoint's tests.
- [Playback measurements and deployment](validation/playback-20260928/RESULTS.md): compact closed benchmark summaries, reusable tests, the initial paused port-18772 verification and the [2.8 MB playback candidate archive](artifacts/rek-native-clone-playback-candidate-r1.tar.gz). This evidence is separate from the original release below.
- [Realtime measurements](validation/realtime-20260928-r1/RESULTS.md): the current single-arena GPU configuration, fixed-deadline scheduling and measured CPU affinity. The final 60 s varied-input/render/recording trial sustained **49.99924 control steps/s after its first 5 s**. Including cold startup, 2,991 real steps advanced 59.82 simulated seconds in 60.007963 wall seconds, or **0.996868x**. The single cold overrun is recorded separately; no simulation steps were fabricated or skipped.
- [Compiled release and evidence](artifacts/native-clone-release-r1.zip): 33,309,978 expanded bytes, including the 7,022,368-byte ARM64/SM121 executable, source, app, closed GPU test results, startup verification and dependency inventory. ZIP SHA256: `fc872376cbf52feab617d255be83b6b1439a5bd8655c810044036e69b52268d7`. Its `PUBLICATION.json` verifies 282 files. The original archive retains its historical four-hour helper; use the separate `passive_support/` revision for continued access.

Source and artifact hashes are preserved without Git newline conversion. Raw, growing human gameplay captures remain on the physical server and are not part of this Git snapshot. External motor weights, shared libraries, Warp PTX modules and reused build objects are inventoried in the release's `dependencies/DEPENDENCIES.json`; they remain at their existing Spark locations. This is a reproducible package against that host dependency closure, not a portable Windows executable or a clean-room build.

Windows checkouts need Git's `core.longpaths=true` because the frozen generated kernel paths retain their original names. The Spark build runs on Linux.

## Build and prepare on Spark

From this directory, after verifying the dependency manifest:

```sh
bash build.sh /absolute/path/to/fresh-native-build
node app/prepare.cjs tools/worker-template.json \
  /absolute/path/to/fresh-native-build/rek-native-clone \
  /absolute/path/to/fresh-native-run tools/env.json 18772
python3 tools/playback/launch_viewer.py --app "$PWD/app" \
  --run /absolute/path/to/fresh-native-run \
  --manifest-sha256 VERIFIED_APP_MANIFEST_SHA256 \
  --binary-sha256 VERIFIED_NATIVE_BINARY_SHA256 \
  --cpus 5,6,7,8,9,15,16,17,18,19
```

Use the explicit verified hashes, an unused port and a fresh run directory. The playback launcher reserves loopback port 18772 and requires an executable supporting `--render-only`; the frozen release-r1 executable predates that protocol. The historical verification run `/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r4` uses port 18771 and independent recording. Preserve it during human play. The new launcher never contacts that port. The tools' historical smoke/launch scripts retain their original experiment paths and fresh-output guards.

The current manual template uses one arena and the verified batch-2 motor exports. Every learned coefficient matches the previous batch-8 exports. Native cross-batch decoder comparisons differed by at most 9.536743e-6, below the existing 2e-5 controller criterion; they were not bit-identical. The timestep, physics model, solver configuration, motor weights and native binary remain unchanged from the playback candidate. CUDA graphs remain disabled. CPU IDs above are the tested Spark allocation, not a portable topology assumption. [Controller and timing evidence](validation/realtime-20260928-r1/RESULTS.md) records these limits.

Revision 7 is deployed in fresh `run-r6` on port 18772, verified paused at tick 0 with its controls, decoded image, process affinities, recording and NAS mirror healthy. The unused prior run-r5 was closed and fully backed up after proving it contained no gameplay. The original port-18771 human session remains preserved. [Deployment receipts](validation/realtime-deployment-20260928-r1/README.md) and the [current publication manifest](publication/realtime-20260928-r1/MANIFEST.json) pin the delivered configuration.

The revision 4 playback candidate keeps each simulation control step at 20 ms, with ten 2 ms physics substeps. Its renderer consumes immutable qpos snapshots at at most 20 Hz with one request in flight and one replaceable latest snapshot. Rendering does not block ordinary play, pause or reset; explicit paused `frame:true` requests still wait for their exact image. Frame tick, reset generation, SHA-256 and publication age are exposed for diagnostics. Source tests alone do not establish deployment, a 1x wall-clock pace or official-game parity.

Historical revision 4 was deployed in run-r5 on port 18772 and verified paused at tick 0 on 2026-09-28 at 21:08:01 UTC, with unrestricted CPU affinity. Its preceding private benchmark measured **0.772844x** real time, compared with the earlier viewer's aggregate observed **0.550308x**. The candidate worker measured **44.882 control ticks/s**. Those earlier workloads do not establish 1x execution or official-game parity. [Historical timing denominators and receipts](validation/playback-20260928/RESULTS.md) remain preserved. Current revision 7 results are linked above.

For a separate headless process, `python3 tools/native_cli.py FRESH_PREPARED_RUN` verifies the file pins and forwards JSON lines. Example request:

```json
{"id":1,"op":"step","humanSide":0,"steps":1,"command":{"forward":0.3,"strafe":-0.7,"yaw":0.2,"moveIndex":4,"cancelAction":false}}
```

The browser also exposes `/api/protocol`, `/api/snapshot`, `/api/reset` and serialized paused `/api/step`. Commands include three continuous axes, an attack edge and cancellation. Replies include 72 positions, 70 velocities, 446 raw observation values, acceptance/rejection, scores, falls and native round/match outcomes. Every manually executed control step is recorded; rendered frames are sampled. Substep contact/force arrays are not fully logged.

## Actual validation

The frozen release-r1 binary SHA256 is `90a004a6aa89a72d13e1c44abe10f2adea70059c746075b062bedb442ed36b73`. The following GPU integration and throughput results describe that release; they do not describe the later playback candidate.

- Ten GPU integration checks passed, including inactive-command rejection without deferred attacks, reset, continuous control, ownership, renderer output and healthy native execution.
- The final Bot1-versus-idle test completed two rounds, both 0:2 from the idle side, and one native Bot1 match win. The result remained latched for 300 additional ticks. These are clone integration observations, not official win rates.
- Headless throughput: 512 ticks in 11.787818447 s, **43.43467 control steps/s per arena**, **173.73868 aggregate arena steps/s across four arenas**. One control tick is 20 ms with ten 2 ms physics substeps; rendering was excluded.
- The release app passed 28 CPU tests. The revision 4 app separately passed [39 CPU tests](app/validation/tests-r4-final.txt), including stalled-renderer progress, reset races, frame identities and renderer timeout isolation. The fresh-port launcher passed [7 mock tests](tools/playback/launcher-tests-r2.txt), including identity mismatch and post-launch failure cleanup. These tests launch no native runtime or viewer. Original history probes matched 27,832 outputs exactly. Match/deactivation and legacy-controller suites passed; their reports and reusable tests accompany the published evidence.
- CUDA graph stepping remains disabled: its strict trajectory comparison failed and the measured speedup was only 1.11x.
- The [original Slerp diagnostic](validation/original-slerp-20260928/RESULTS.md) reproduced all 640 composer fixture rows when the Slerp callback used measured original-function outputs. All 12,800 quaternion components, 92,800 joint references and other supported state fields matched exactly. The final replay covered every one of 32,800 calls without a fallback. This is an input-specific diagnostic; the deployed native math remains unchanged.
- The [original Unity serialized asset inventory](validation/original-model-20260928/rek-original-model-r2/RESULT.md) completed with 226 transforms and 498 components, including 30 bodies, 30 inertials, 29 hinges, one free joint, 37 geoms and 29 actuators. It captured two robot configurations, 36 motion clip configurations and the loaded Arena global settings. Its 201 serialized JSON records are preserved with hashes. The first failed interop attempt and its ABI diagnosis remain alongside the successful retry. These are asset observations; effective compiled options and initialized controller state still require measurement.

The composer fixture's earlier quaternion residual is isolated to the Slerp callback boundary. A general native Slerp replacement remains to be implemented and verified. The next environment comparison is the [original compiled model and initialized/reset state](validation/parity-priorities-20260928/NEXT-EXPERIMENT.md). Prior whole-trajectory yaw disagreement has no new matched official replay proving resolution. Lighting/materials differ from Unity; the renderer retains a documented setup warning despite successful decoded PNG verification. Full environment parity remains unverified.
