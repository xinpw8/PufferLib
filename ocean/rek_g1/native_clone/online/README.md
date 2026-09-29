# REK Online

Two-player online version of the native REK viewer, behind a shared password.

- One player logged in: they wait in the lobby and can press **Fight Bot 1** (the recovered physical Bot 1 on fighter 2).
- A second player logs in: any Bot 1 match ends and a player-versus-player match starts automatically after a 3 s countdown. After a match ends, the next one starts 8 s later while both stay.
- Further visitors watch; they take a seat when a player leaves.

## Why it is responsive

The previous viewer rendered PNG frames on the server (17 ms each, 400–580 KB) and the browser polled them over HTTP, with keys sent as separate HTTP posts. Here the server never renders:

- One authoritative native simulation steps at 50 Hz (20 ms control ticks). After every step the server broadcasts a 328-byte binary state (72 joint positions plus score, clock and command feedback) over WebSocket, about 16 KB/s per player.
- Each browser renders the arena itself with WebGL (three.js) from a one-time scene export (`export_scene`), computing robot poses from the joint positions exactly as MuJoCo does (`public/scene.js`; tested against `mj_kinematics` to 1e-8 m). It interpolates between ticks on a small jitter-adaptive buffer (25 ms minimum), and each player has their own follow camera.
- Keys go over the same socket and reach the next simulation tick.

Measured on Spark with the server at 50.0 ticks/s (step p50 16 ms, p99 22 ms): attack key to native acceptance at the client was 17–32 ms locally and 32–55 ms through a Cloudflare tunnel (14 ms ping). Internet latency between each player and Spark adds to that.

## Pieces

- `server.cjs`: HTTP, password login, WebSocket, lobby and the 50 Hz loop. Reuses `app/league/worker.cjs`, `input.cjs`, `paced_loop.cjs` and `human_session.cjs` unchanged.
- `auth.cjs`: scrypt password hash, HMAC-signed 7-day cookie, per-address login limiter (8 failures per 15 min).
- `lobby.cjs`: seats and mode rules. `protocol.cjs`: tick message layout.
- `public/`: login page, game page and client. `vendor/ws` and `public/vendor/three.module.min.js` are vendored (MIT).
- `export_scene.cpp`: writes `scene.json`, `scene.bin` and MuJoCo's decoded textures for the presentation model.
- Native worker: `step` accepts `commands: [fighter0, fighter1]` for two human players with no scripted opponent. Single-command and policy paths are unchanged; switching needs a reset, which the server does for every new match.

## Build and run

```sh
# Native worker (revision-9 recipe: reuses the build-r6 objects, recompiles eval_worker.cpp)
REK_ONLINE_BUILD_DIR=/path/to/fresh-build bash online/build-worker.sh
# Scene export for the browser
M=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
g++ -std=c++17 -O2 -I$M/include online/export_scene.cpp -L$M -l:libmujoco.so.3.7.0 -lz -Wl,-rpath,$M -o export_scene
./export_scene app/presentation/assets/presentation.playable.xml SCENE_DIR
# Private config with a new random password (printed to PASSWORD_FILE, mode 0600)
node online/configure.cjs --out CONFIG.json --password-file PASSWORD_FILE --binary BINARY --worker-config WORKER.json \
  --server-json RUN/server.json --scene-dir SCENE_DIR --log-dir LOG_DIR --port 18780 --hosts rek.clipfrac.com
online/start.sh CONFIG.json RUN_DIR
# Change the password later (reads it from stdin; --logout-all also signs everyone out), then restart
node online/set_password.cjs CONFIG.json [--logout-all]
```

The server listens on 127.0.0.1 only; a Cloudflare tunnel publishes it at `https://rek.clipfrac.com`. Only allowed Host names are served, and WebSocket upgrades require the login cookie and a same-origin `Origin`. Logs are small (joins, matches) and capped; `disk_cap.py` stops the server if its log directory passes 2 GB.

Tests: `node --test test/server.test.cjs` (auth, lobby, protocol, login → WebSocket → Bot 1 → PvP flow with a fake worker) and `node --test test/kinematics.test.mjs SCENE_DIR QPOS.f32` (browser kinematics against MuJoCo reference poses written by `export_scene MODEL SCENE_DIR QPOS.f32`).
