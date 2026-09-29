# REK Native Clone app

Revision 10 replaces the overhead presentation camera with a third-person view behind the human-controlled fighter. The server adds `followSide`, the active human side, to every frame snapshot; the render-only worker echoes it and places the camera 3 m behind the fighter's pelvis, 1 m above it and tilted 10 degrees down. These are the recovered `RobotFollowCamera` constructor defaults; REK's scene overrides, active camera mode and FOV remain unverified, and the model's 45-degree MuJoCo FOV is unchanged. Heading uses the recovered `CalcHeadingMj` operand order. The view holds inside a 20-degree dead band, trails larger turns at the band edge and recentres with a one-second time constant measured in snapshot ticks, so render pacing cannot change it; a new generation, side or earlier tick snaps it. A downed fighter keeps the last heading. On the recorded perf-graph trajectory a 14.8-degree strike swing of the pelvis moved the view 3.1 degrees, and the largest step between rendered snapshots fell from 3.09 to 1.17 degrees. Arena meshes whose wall plane the eye has crossed are dropped from that frame's scene, so the cage never blocks the fighter; far walls, the floor and both fighters remain. Requests without `followSide` render the unchanged overview byte for byte, and an older native binary ignores the field and keeps the overview. Collision geometry, physics, controls and recorded states are unchanged. `presentation/validate_follow_camera.cpp` checks request parsing, the heading filter and overview isolation, then writes follow frames for inspection; it links MuJoCo, EGL, zlib and the vendored cJSON source.

Revision 8 corrects the manual attack translation. Saved keyboard gestures and buttons still emit policy categories 16 through 32, while native direct commands receive the original move-registry indices. The four kick categories now map to native indices 6 through 9; the six preceding punch entries map to 0 through 5. Named tests join every saved command to the pinned original clip and route instead of assuming `category - 16`. HH selects left front kick, UU selects right side kick, and L selects right jab. Gesture timing, continuous controls, native acceptance/rejection, physics and pacing are unchanged. Native busy rejection remains possible for a correctly named request.

Revision 7 uses absolute monotonic 20 ms deadlines. Callback lateness does not shift every later deadline. It keeps one task and one scheduled callback, waits again after an early wake, and cancels/rebases scheduling on pause/resume and reset. Wall-clock debt above 100 ms is explicitly discarded and counted; no physics step, input/state record or timestep is changed. Scheduler task cost, lateness and rebases are exposed separately from the unchanged measured simulation/wall-time ratio. Actual 1x execution still requires the private integrated benchmark.

Revision 6 adds an explicitly bound one-arena configuration. It accepts only the existing batch-2 encoder SHA256 `0e1cf37a7c1bafe870741b8a3de2560ae7a2b0cfe14c5ac8cef4d8a34d78207f` and decoder SHA256 `20b49c9df1a54dc3a211d0d86c2ebe3ccc1de67883a984a7b227af76af7aacb3`. The established four-arena path remains available. Preparation retains GPU/full-physics/Bot1 guards and records arena count, required controller batch, displayed arena, fixed control dt and the aggregate arena-step multiplier. One control tick advances one arena in the one-arena configuration; the old four-arena aggregate throughput multiplier does not apply. Controller and app runtime comparisons remain separate evidence.

Revision 5 is a separate performance candidate. Overdue control work yields with `setImmediate` instead of a zero-delay timer, avoiding the timer's minimum delay while retaining one request at a time and no catch-up backlog. Input packets, resolved commands and every native request/reply remain synchronously recorded. Ordinary step health-file updates are limited to one per 250 ms; reset, snapshot, error and exit updates are immediate. Recorder and server share one decoded PNG and SHA-256 through a weak cache, without changing the 20 Hz renderer limit or image resolution. No runtime throughput or deployment result is established by these source changes.

Revision 4 runs presentation in a separate `--render-only` process. The physics loop sends one fixed 20 ms step at a time and never waits for image generation. Presentation receives immutable 72-value qpos snapshots at at most 20 Hz, with one rendering request and one replaceable latest snapshot retained. Replies must echo the source tick and reset generation. Reset clears the old image and rejects old-generation results. Renderer errors are reported separately and do not stop physics. Each published image has a SHA-256, source tick, generation and publication age; archived frames preserve those identities. This removes serial rendering overhead; measured real-time speed still depends on the native step cost and resource contention.

Revision 3 displays reason 4 as "round inactive", including rejected velocity-only commands. Reason 4 (`INPUT_INACTIVE`) is a clone telemetry extension, not a claim about the original game's numeric reason values. The native runtime enforces the round-phase input gate. Browser queueing and action-mask behavior are unchanged.

Revision 2 preserves the original app and uses `intermissionMs:0` in its launched server. The native five-second between-round phase is the sole intermission timer. Countdown, fighting, between-rounds and match-complete labels come directly from the worker phase enum; the browser does not invent a countdown duration. Match statistics rules are unchanged. The renderer header includes the separately tested camera elevation of -60 degrees, retaining all arena geometry and both robots' full-body visibility at the tested initial pose.

This isolated app uses the full native MuJoCo motor evaluator, the recovered Bot1 strategy, original exported visual mesh instances, and Daniel's saved G1 bindings. Full official simulation parity remains unverified. The native executable and its configuration are supplied by the parent build; this package does not start or change any existing game or viewer.

## Prepare and run on Spark

Use a fresh run directory, the verified native clone binary, a full four-arena worker template, and an explicit JSON environment map:

```sh
node /path/to/app/prepare.cjs /path/to/worker-template.json /path/to/current-build/rek-native-clone /path/to/fresh-run /path/to/environment.json 18772
bash /path/to/app/start.sh /path/to/fresh-run
```

The service binds only to loopback on the supplied port and starts paused. An SSH tunnel can expose that port to the user's browser. The environment must select `mujoco_cuda`, disable CPU evaluation, select `recovered_bot1_g1_v1`, and set `REK_MUJOCO_DETERMINISTIC_SCALAR=0`. Existing viewers and the official client stay separate. One physics worker with one verified batch-2 arena or the existing four batch-8 arenas, plus one presentation worker with no dynamics/controller construction, forms the manual configuration. The executable must support the revision 4 renderer protocol.

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

An explicit paused `frame:true` step still waits for its exact resulting image. Ordinary play, pause and reset do not wait for rendering. `/frame.png` exposes `X-Rek-Snapshot-Tick`, `X-Rek-Frame-Generation` and `X-Rek-Frame-Sha256`; `/api/state` exposes `frame` and `renderFailure`.

`launch_logged.cjs` writes input packets, resolved commands, every native request/reply, reset boundaries and rendered PNG frames with UTC, monotonic timestamps and hashes. These are native-clone human trajectories, not authoritative official-game data. Run outputs are append-only and separate from package assets. The package performs no training and makes no throughput claim.

## Validation

```sh
node --test league/*.test.cjs
```

The focused tests cover saved controls, continuous command edges, serial pacing, pause/reset lifecycle, native outcome bookkeeping, ownership rejection, delayed-body request races, stalled rendering, latest-snapshot coalescing, stale-generation rejection, exact paused frames and renderer timeout isolation. Revision 5 adds timer-clamp cancellation, health-write failure propagation, shared image identity and the actual logging wrapper against a fake worker. Its logging test preserves 50 input/request/state triples and checks frame bytes, error status and reset status without a native process. `validation/tests-r5.txt` records the current CPU run; `validation/tests-r4-final.txt` remains the parent receipt. Native rendering, actual throughput and visual usability require separate integrated measurement. `freeze_r5.py` preserves `SOURCE-MANIFEST.parent-r4.json` and creates this revision's fresh manifest/archive; older freeze scripts remain historical source.
