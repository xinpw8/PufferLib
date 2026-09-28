# REK Native Clone app

Revision 3 displays reason 4 as "round inactive", including rejected velocity-only commands. Reason 4 (`INPUT_INACTIVE`) is a clone telemetry extension, not a claim about the original game's numeric reason values. The native runtime enforces the round-phase input gate. Browser queueing and action-mask behavior are unchanged.

Revision 2 preserves the original app and uses `intermissionMs:0` in its launched server. The native five-second between-round phase is the sole intermission timer. Countdown, fighting, between-rounds and match-complete labels come directly from the worker phase enum; the browser does not invent a countdown duration. Match statistics rules are unchanged. The renderer header includes the separately tested camera elevation of -60 degrees, retaining all arena geometry and both robots' full-body visibility at the tested initial pose.

This isolated app uses the full native MuJoCo motor evaluator, the recovered Bot1 strategy, original exported visual mesh instances, and Daniel's saved G1 bindings. Full official simulation parity remains unverified. The native executable and its configuration are supplied by the parent build; this package does not start or change any existing game or viewer.

## Prepare and run on Spark

Use a fresh run directory, the verified native clone binary, a full four-arena worker template, and an explicit JSON environment map:

```sh
node /path/to/app/prepare.cjs /path/to/worker-template.json /path/to/build-r2/rek-native-clone /path/to/fresh-run /path/to/environment.json 18771
bash /path/to/app/start.sh /path/to/fresh-run
```

The service binds only `127.0.0.1:18771` and starts paused. An SSH tunnel can expose that port to the user's browser. The environment must select `mujoco_cuda`, disable CPU evaluation, select `recovered_bot1_g1_v1`, and set `REK_MUJOCO_DETERMINISTIC_SCALAR=0`. Existing viewers and the official client stay separate. One worker, four arenas, and two OpenMP threads are the intended manual configuration.

The prepared identity pins the executable, worker/environment inputs, physics/model and motor files, render XML and saved controls. `SOURCE-MANIFEST.json` pins every packaged source and asset. The native build must include `presentation/eval_renderer.h`, which renders the distinct `render_model_path`; it never steps presentation physics. The collision XML, mesh vertices, original link transforms and native physics are unchanged by this app. The camera and headlight are presentation choices. Original shaders, all material/submesh behavior and photometric equality remain unverified.

## Play

Click the arena or Play. WASD moves, Q/E turns, and the listed attack buttons reproduce the saved keyboard profile, including Space chords and double taps. A is positive strafe; D is negative strafe in the recovered control convention. Diagonals retain both unit components. Keyboard input is scoped to the focused arena. Blur releases held controls; a hidden/unfocused page pauses play. Escape pauses. New match explicitly resets native match state and records an unfinished prior match separately. A completed native match stays paused until reset.

The native scheduler receives movement and attacks together and decides acceptance. The browser does not suppress attacks using action masks or busy-state estimates. Scores, falls, round results and match winners come from native state. Session match wins are counted only from a native match terminal, never inferred from points or two round wins.

Pacing schedules the next fixed 20 ms simulation step after work completes and the period permits. It avoids quantizing 24 ms work into 40 ms intervals, without scaling simulation time or running concurrent native requests. The UI reports measured active-span pace and request/render cost. Real-time performance requires the actual integrated smoke/play measurement.

## Paused training protocol

`GET /api/protocol` describes dimensions. `GET /api/snapshot` reads state without renewing the manual client heartbeat. `POST /api/reset` clears native match/ownership state and remains paused. `POST /api/step` requires a paused, ready worker and exactly one `command` or categorical `action`:

```json
{"steps":1,"command":{"forward":0.3,"strafe":-0.7,"yaw":0.2,"moveIndex":4,"cancelAction":false},"frame":false}
```

Steps accepts 1 through 512 and stops at a native round boundary. Continuous components remain in [-1,1]. `moveIndex` accepts -1 for none and 0 through 16 for one attack edge on the first step only. Native `commandEvents` preserve each attempted/accepted/rejected event. Replies retain raw observations, masks, qpos/qvel and native round/match metrics. Explicit reset is required to switch between direct and categorical ownership. Native rejection is recorded separately from recorder failure.

`launch_logged.cjs` writes input packets, resolved commands, every native request/reply, reset boundaries and rendered PNG frames with UTC, monotonic timestamps and hashes. These are native-clone human trajectories, not authoritative official-game data. Run outputs are append-only and separate from package assets. The package performs no training and makes no throughput claim.

## Validation

```sh
node --test league/input.test.cjs league/controls.test.cjs league/paced_loop.test.cjs league/human_session.test.cjs league/server.test.cjs league/protocol.test.cjs league/native_phase.test.cjs
```

The focused tests cover saved controls, continuous command edges, serial pacing, pause/reset lifecycle, native outcome bookkeeping, ownership rejection and delayed-body request races. `validation/tests.txt` records the package test run. Native rendering, the actual GPU executable and visual usability require the parent integrated smoke test.
