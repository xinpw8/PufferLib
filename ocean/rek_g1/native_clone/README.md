# Native REK clone and measured evidence

This package preserves the working C++/CUDA clone tested on DGX Spark. It includes the exact source overlay, the browser app and original exported meshes, saved G1 controls, build and test tools, passive transport/backup helpers, and the compiled release with its verification records.

**Full official-game parity remains unverified.** Recovered controller, match and referee functions have targeted tests; authoritative physics, initial state, global callback order, server assets and end-to-end trajectories still require matched official comparisons. This package does not establish superhuman play or a successful learned policy.

## Contents

- `source/`: exact 50-file source closure used for build-r6. Relative includes are preserved. The shared legacy evaluator remains separate because its reduced backend does not implement the new direct-command API.
- `dependency_source/`: frozen historical source files associated with the inventoried build dependencies, with original paths and hashes. These preserve the source record; the current build still consumes the pinned Spark objects and libraries.
- `app/`: paused-by-default playable viewer, continuous commands, original mesh rendering, saved controls, native round/match metrics and detailed recording.
- `tools/`: pinned build recipe, worker/environment templates, GPU smoke/match test drivers and headless JSON-line launcher.
- `passive_support/`: loopback SSH tunnel and append-verified NAS mirror. Explicit `--until-stop` removes the old four-hour expiry; omitting it retains the timed default.
- `validation/`: closed test results, original-function comparisons, dependency inventories, connection repair evidence and the preceding project checkpoint's tests.
- [Compiled release and evidence](artifacts/native-clone-release-r1.zip): 33,309,978 expanded bytes, including the 7,022,368-byte ARM64/SM121 executable, source, app, closed GPU test results, startup verification and dependency inventory. ZIP SHA256: `fc872376cbf52feab617d255be83b6b1439a5bd8655c810044036e69b52268d7`. Its `PUBLICATION.json` verifies 282 files. The original archive retains its historical four-hour helper; use the separate `passive_support/` revision for continued access.

Source and artifact hashes are preserved without Git newline conversion. Raw, growing human gameplay captures remain on the physical server and are not part of this Git snapshot. External motor weights, shared libraries, Warp PTX modules and reused build objects are inventoried in the release's `dependencies/DEPENDENCIES.json`; they remain at their existing Spark locations. This is a reproducible package against that host dependency closure, not a portable Windows executable or a clean-room build.

## Build and prepare on Spark

From this directory, after verifying the dependency manifest:

```sh
bash build.sh /absolute/path/to/fresh-native-build
node app/prepare.cjs tools/worker-template.json \
  /absolute/path/to/fresh-native-build/rek-native-clone \
  /absolute/path/to/fresh-native-run tools/env.json 18771
node app/launch_logged.cjs /absolute/path/to/fresh-native-run
```

Use an unused port and a fresh run directory. The currently deployed verification run is `/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r4`. It serves port 18771 and has independent recording; do not replace or reset it during human play. The tools' historical smoke/launch scripts retain their original experiment paths and fresh-output guards.

For a separate headless process, `python3 tools/native_cli.py FRESH_PREPARED_RUN` verifies the file pins and forwards JSON lines. Example request:

```json
{"id":1,"op":"step","humanSide":0,"steps":1,"command":{"forward":0.3,"strafe":-0.7,"yaw":0.2,"moveIndex":4,"cancelAction":false}}
```

The browser also exposes `/api/protocol`, `/api/snapshot`, `/api/reset` and serialized paused `/api/step`. Commands include three continuous axes, an attack edge and cancellation. Replies include 72 positions, 70 velocities, 446 raw observation values, acceptance/rejection, scores, falls and native round/match outcomes. Every manually executed control step is recorded; rendered frames are sampled. Substep contact/force arrays are not fully logged.

## Actual validation

The selected binary SHA256 is `90a004a6aa89a72d13e1c44abe10f2adea70059c746075b062bedb442ed36b73`.

- Ten GPU integration checks passed, including inactive-command rejection without deferred attacks, reset, continuous control, ownership, renderer output and healthy native execution.
- The final Bot1-versus-idle test completed two rounds, both 0:2 from the idle side, and one native Bot1 match win. The result remained latched for 300 additional ticks. These are clone integration observations, not official win rates.
- Headless throughput: 512 ticks in 11.787818447 s, **43.43467 control steps/s per arena**, **173.73868 aggregate arena steps/s across four arenas**. One control tick is 20 ms with ten 2 ms physics substeps; rendering was excluded.
- The app passed 28 CPU tests. Original history probes matched 27,832 outputs exactly. Match/deactivation and legacy-controller suites passed; their reports and reusable tests accompany the published evidence.
- CUDA graph stepping remains disabled: its strict trajectory comparison failed and the measured speedup was only 1.11x.

Earlier original-composer quaternion differences at the Unity Slerp comparison boundary are still under investigation. Prior whole-trajectory yaw disagreement has no new matched official replay proving resolution. Lighting/materials differ from Unity; the renderer retains a documented setup warning despite successful decoded PNG verification. Current work continues with isolated original-function oracles before any new parity claim.
