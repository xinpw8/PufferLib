#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import struct
import threading
import time
import urllib.parse
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from engineai_t800_policy import (
    CONTROL_DT,
    T800MuJoCoBinding,
    T800SupineRecoveryController,
    T800WalkingController,
)


INDEX_HTML = r"""
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>REK Fight Human Eval</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { margin: 0; background: #0d1117; color: #e6edf3; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 20px; font-weight: 500; margin: 0 0 10px; }
  .notice { color: #f0c36a; margin: 0 0 12px; }
  .viewport { position: relative; background: #05070a; border: 1px solid #30363d; }
  #frame { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; }
  .hud { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }
  .hud div { background: #161b22; padding: 8px 10px; border-radius: 6px; }
  .label { display: block; color: #8b949e; font-size: 12px; }
  .controls { line-height: 1.7; }
  kbd { background: #21262d; border: 1px solid #30363d; border-bottom-width: 2px; border-radius: 4px; padding: 1px 6px; }
  button { background: #238636; color: white; border: 0; border-radius: 6px; padding: 9px 14px; cursor: pointer; }
  #connection { margin-left: 10px; color: #8b949e; }
  @media (max-width: 620px) { .hud { grid-template-columns: repeat(2, 1fr); } }
</style>
<main>
  <h1>REK Fight Human Eval</h1>
  <p class="notice">Official EngineAI T800 walking policy. Blue is you. Orange uses the same walking policy as a deterministic approach dummy. REK combat moves are disabled until their trajectories are measured.</p>
  <div class="viewport"><img id="frame" alt="Live REK fight simulator view"></div>
  <div class="hud" aria-live="polite">
    <div><span class="label">Tick</span><span id="tick">0</span></div>
    <div><span class="label">Your hits</span><span id="humanHits">0</span></div>
    <div><span class="label">Dummy hits</span><span id="botHits">0</span></div>
    <div><span class="label">Return</span><span id="return">0.000</span></div>
  </div>
  <p class="controls">
    <kbd>W</kbd>/<kbd>S</kbd> forward/back,
    <kbd>A</kbd>/<kbd>D</kbd> strafe,
    <kbd>Q</kbd>/<kbd>E</kbd> yaw,
    Combat keys are disabled in this controller-validation build.
    Click this page once before using the keyboard.
  </p>
  <button id="reset" type="button">Reset round</button><span id="connection">connecting</span>
</main>
<script>
(() => {
  const held = new Set();
  const relevant = new Set(['KeyW','KeyS','KeyA','KeyD','KeyQ','KeyE']);
  let inputSequence = 0;
  const axis = (negative, positive) => held.has(negative) === held.has(positive) ? 1 : held.has(positive) ? 2 : 0;
  async function sendInput() {
    const payload = {
      sequence: ++inputSequence,
      forward: axis('KeyS', 'KeyW'),
      strafe: axis('KeyD', 'KeyA'),
      yaw: axis('KeyE', 'KeyQ'),
      move: 0
    };
    try {
      await fetch('/input', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
    } catch (_) {}
  }
  addEventListener('keydown', event => {
    if (!relevant.has(event.code)) return;
    event.preventDefault();
    held.add(event.code);
    sendInput();
  });
  addEventListener('keyup', event => {
    if (!relevant.has(event.code)) return;
    event.preventDefault();
    held.delete(event.code);
    sendInput();
  });
  addEventListener('blur', () => { held.clear(); sendInput(); });
  document.getElementById('reset').addEventListener('click', async () => {
    held.clear();
    await fetch('/reset', {method:'POST'});
  });
  const image = document.getElementById('frame');
  function nextFrame() { image.src = '/frame.png?t=' + Date.now(); }
  image.addEventListener('load', () => setTimeout(nextFrame, 20));
  image.addEventListener('error', () => setTimeout(nextFrame, 500));
  nextFrame();
  async function pollState() {
    try {
      const state = await (await fetch('/state')).json();
      document.getElementById('tick').textContent = state.tick;
      document.getElementById('humanHits').textContent = state.hits[0];
      document.getElementById('botHits').textContent = state.hits[1];
      document.getElementById('return').textContent = state.episode_return[0].toFixed(3);
      document.getElementById('connection').textContent = 'live at 100 Hz control / 500 Hz physics';
    } catch (_) {
      document.getElementById('connection').textContent = 'disconnected';
    }
    setTimeout(pollState, 100);
  }
  pollState();
})();
</script>
"""


def png_bytes(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected HxWx3 uint8 RGB")
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 3))
        + chunk(b"IEND", b"")
    )


class HumanEval:
    def __init__(self, library_path: Path, model_path: Path, log_path: Path | None):
        self.library = ctypes.CDLL(str(library_path))
        self.library.rek_human_create.restype = ctypes.c_void_p
        self.library.rek_human_destroy.argtypes = [ctypes.c_void_p]
        self.library.rek_human_reset.argtypes = [ctypes.c_void_p]
        self.library.rek_human_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.rek_human_nq.argtypes = [ctypes.c_void_p]
        self.library.rek_human_nq.restype = ctypes.c_int
        self.library.rek_human_nv.argtypes = [ctypes.c_void_p]
        self.library.rek_human_nv.restype = ctypes.c_int
        self.library.rek_human_copy_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        for name in ("tick", "hits", "fallen", "move_slot"):
            function = getattr(self.library, f"rek_human_{name}")
            function.argtypes = [ctypes.c_void_p] if name == "tick" else [ctypes.c_void_p, ctypes.c_int]
            function.restype = ctypes.c_int
        for name in ("reward", "episode_return"):
            function = getattr(self.library, f"rek_human_{name}")
            function.argtypes = [ctypes.c_void_p, ctypes.c_int]
            function.restype = ctypes.c_float

        self.session = self.library.rek_human_create()
        if not self.session:
            raise RuntimeError("rek_human_create failed")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            if 1 <= body <= 30:
                self.model.geom_rgba[geom] = (0.10, 0.65, 0.95, 1.0)
            elif 31 <= body <= 60:
                self.model.geom_rgba[geom] = (0.95, 0.40, 0.12, 1.0)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=360, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 6.2
        self.camera.azimuth = 90.0
        self.camera.elevation = -42.0
        self.nq = self.library.rek_human_nq(self.session)
        self.nv = self.library.rek_human_nv(self.session)
        if self.nq != self.model.nq or self.nv != self.model.nv:
            raise RuntimeError("bridge and renderer model dimensions differ")
        self.qpos = np.zeros(self.nq, dtype=np.float64)
        self.qvel = np.zeros(self.nv, dtype=np.float64)
        self.action = {"forward": 1, "strafe": 1, "yaw": 1, "move": 0, "sequence": 0}
        self.pending_move = 0
        self.lock = threading.RLock()
        self.running = True
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="rek-human-eval-step", daemon=True)
        self.thread.start()

    def _write_log(self, kind: str, payload: dict) -> None:
        if not self.log_path:
            return
        record = {"time_unix_ns": time.time_ns(), "kind": kind, **payload}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _run(self) -> None:
        period = 0.02
        deadline = time.monotonic()
        while self.running:
            deadline += period
            with self.lock:
                action = self.action.copy()
                action["move"] = self.pending_move
                self.pending_move = 0
                self.library.rek_human_step(
                    self.session,
                    action["forward"],
                    action["strafe"],
                    action["yaw"],
                    action["move"],
                )
                if action["move"]:
                    self._write_log("move", {"tick": self.tick(), "action": action})
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()

    def tick(self) -> int:
        return self.library.rek_human_tick(self.session)

    def update_action(self, payload: dict) -> None:
        parsed = {
            "forward": max(0, min(2, int(payload.get("forward", 1)))),
            "strafe": max(0, min(2, int(payload.get("strafe", 1)))),
            "yaw": max(0, min(2, int(payload.get("yaw", 1)))),
            "move": max(0, min(6, int(payload.get("move", 0)))),
            "sequence": int(payload.get("sequence", 0)),
        }
        with self.lock:
            if parsed["sequence"] < self.action["sequence"]:
                return
            self.action = parsed
            if parsed["move"]:
                self.pending_move = parsed["move"]
            self._write_log("input", {"tick": self.tick(), "action": parsed})

    def reset(self) -> None:
        with self.lock:
            self.library.rek_human_reset(self.session)
            self.action = {"forward": 1, "strafe": 1, "yaw": 1, "move": 0, "sequence": self.action["sequence"]}
            self.pending_move = 0
            self._write_log("reset", {"tick": self.tick()})

    def state(self) -> dict:
        with self.lock:
            return {
                "tick": self.tick(),
                "hits": [self.library.rek_human_hits(self.session, agent) for agent in range(2)],
                "fallen": [self.library.rek_human_fallen(self.session, agent) for agent in range(2)],
                "move_slot": [self.library.rek_human_move_slot(self.session, agent) for agent in range(2)],
                "reward": [float(self.library.rek_human_reward(self.session, agent)) for agent in range(2)],
                "episode_return": [float(self.library.rek_human_episode_return(self.session, agent)) for agent in range(2)],
                "input_sequence": self.action["sequence"],
                "provisional": True,
                "opponent": "deterministic_sparring_dummy",
            }

    def frame(self) -> bytes:
        with self.lock:
            copied = self.library.rek_human_copy_state(
                self.session,
                self.qpos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.nq,
                self.qvel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.nv,
            )
            if copied != 1:
                raise RuntimeError("state copy failed")
            self.data.qpos[:] = self.qpos
            self.data.qvel[:] = self.qvel
            mujoco.mj_forward(self.model, self.data)
            roots = np.vstack((self.qpos[0:3], self.qpos[32:35]))
            self.camera.lookat[:] = roots.mean(axis=0)
            self.camera.lookat[2] = max(0.8, self.camera.lookat[2])
            self.renderer.update_scene(self.data, camera=self.camera)
            return png_bytes(self.renderer.render())

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.renderer.close()
        self.library.rek_human_destroy(self.session)


class PolicyHumanEval:
    def __init__(
        self,
        model_path: Path,
        policy_path: Path,
        recovery_policy_path: Path,
        recovery_trajectory_path: Path,
        log_path: Path | None,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.source_timestep = float(self.model.opt.timestep)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.bindings = [T800MuJoCoBinding(mujoco, self.model, fighter=agent) for agent in range(2)]
        self.controllers = [T800WalkingController(policy_path) for _ in range(2)]
        self.recovery_controllers = [
            T800SupineRecoveryController(recovery_policy_path, recovery_trajectory_path) for _ in range(2)
        ]
        self.recovering = [False, False]
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            if 1 <= body <= 30:
                self.model.geom_rgba[geom] = (0.10, 0.65, 0.95, 1.0)
            elif 31 <= body <= 60:
                self.model.geom_rgba[geom] = (0.95, 0.40, 0.12, 1.0)
        self.renderer = mujoco.Renderer(self.model, height=360, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 6.2
        self.camera.azimuth = 90.0
        self.camera.elevation = -42.0
        ratio = CONTROL_DT / float(self.model.opt.timestep)
        self.substeps = int(round(ratio))
        if abs(ratio - self.substeps) > 1e-9:
            raise ValueError("walking control period is not divisible by the model timestep")
        self.action = {"forward": 1, "strafe": 1, "yaw": 1, "move": 0, "sequence": 0}
        self.lock = threading.RLock()
        self.running = True
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tick_count = 0
        self.reset_count = 0
        self._reset_unlocked()
        self.thread = threading.Thread(target=self._run, name="rek-official-policy-step", daemon=True)
        self.thread.start()

    def _write_log(self, kind: str, payload: dict) -> None:
        if not self.log_path:
            return
        record = {"time_unix_ns": time.time_ns(), "kind": kind, **payload}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _reset_unlocked(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for binding, controller in zip(self.bindings, self.controllers):
            binding.set_default_pose(self.data)
            controller.reset()
        self.recovering[:] = [False, False]
        mujoco.mj_forward(self.model, self.data)
        self.action.update({"forward": 1, "strafe": 1, "yaw": 1, "move": 0})
        self.tick_count = 0
        self.reset_count += 1

    def _human_command(self) -> np.ndarray:
        return np.array(
            [self.action["forward"] - 1, self.action["strafe"] - 1, self.action["yaw"] - 1],
            dtype=np.float64,
        )

    def _bot_command(self) -> np.ndarray:
        bot = self.bindings[1]
        delta_world = self.bindings[0].root_position(self.data) - bot.root_position(self.data)
        quaternion = self.data.qpos[bot.root_qpos_address + 3 : bot.root_qpos_address + 7]
        w, x, y, z = quaternion
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)
        forward = c * delta_world[0] + s * delta_world[1]
        lateral = -s * delta_world[0] + c * delta_world[1]
        distance = float(np.hypot(forward, lateral))
        return np.array(
            [np.clip((distance - 1.2) * 0.8, -0.4, 0.7), np.clip(lateral * 0.5, -0.4, 0.4), 0.0],
            dtype=np.float64,
        )

    def _run(self) -> None:
        deadline = time.monotonic()
        while self.running:
            deadline += CONTROL_DT
            with self.lock:
                commands = [self._human_command(), self._bot_command()]
                targets = []
                for agent, (binding, controller, recovery, normalized_command) in enumerate(
                    zip(self.bindings, self.controllers, self.recovery_controllers, commands)
                ):
                    joint_q, joint_qd, quaternion, angular_velocity = binding.state(self.data)
                    fallen = binding.root_position(self.data)[2] < 0.65 or binding.root_up_z(self.data) < 0.45
                    if fallen and not self.recovering[agent]:
                        recovery.reset(joint_q)
                        self.recovering[agent] = True
                        self._write_log("supine_recovery_start", {"tick": self.tick_count, "agent": agent})
                    if self.recovering[agent]:
                        target, _ = recovery.step(joint_q, joint_qd, quaternion, angular_velocity)
                        targets.append((target, recovery))
                    else:
                        controller.observe(joint_q, joint_qd, quaternion, angular_velocity)
                        _, target = controller.act(controller.scale_command(normalized_command))
                        targets.append((target, controller))
                for _ in range(self.substeps):
                    for binding, (target, controller) in zip(self.bindings, targets):
                        joint_q, joint_qd, _, _ = binding.state(self.data)
                        binding.apply_torque(self.data, controller.pd_torque(joint_q, joint_qd, target))
                    mujoco.mj_step(self.model, self.data)
                for agent, (binding, controller, recovery) in enumerate(
                    zip(self.bindings, self.controllers, self.recovery_controllers)
                ):
                    if self.recovering[agent] and recovery.finished:
                        upright = binding.root_position(self.data)[2] > 0.8 and binding.root_up_z(self.data) > 0.9
                        if upright:
                            self.recovering[agent] = False
                            controller.reset()
                            self._write_log("supine_recovery_complete", {"tick": self.tick_count, "agent": agent})
                self.tick_count += 1
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()

    def tick(self) -> int:
        return self.tick_count

    def update_action(self, payload: dict) -> None:
        parsed = {
            "forward": max(0, min(2, int(payload.get("forward", 1)))),
            "strafe": max(0, min(2, int(payload.get("strafe", 1)))),
            "yaw": max(0, min(2, int(payload.get("yaw", 1)))),
            "move": 0,
            "sequence": int(payload.get("sequence", 0)),
        }
        with self.lock:
            if parsed["sequence"] < self.action["sequence"]:
                return
            self.action = parsed
            self._write_log("input", {"tick": self.tick_count, "action": parsed})

    def reset(self) -> None:
        with self.lock:
            sequence = self.action["sequence"]
            self._reset_unlocked()
            self.action["sequence"] = sequence
            self._write_log("reset", {"tick": self.tick_count, "reset_count": self.reset_count})

    def state(self) -> dict:
        with self.lock:
            fallen = [
                bool(binding.root_position(self.data)[2] < 0.65 or binding.root_up_z(self.data) < 0.45)
                for binding in self.bindings
            ]
            return {
                "tick": self.tick_count,
                "hits": [0, 0],
                "fallen": fallen,
                "move_slot": [0, 0],
                "reward": [0.0, 0.0],
                "episode_return": [0.0, 0.0],
                "input_sequence": self.action["sequence"],
                "provisional": True,
                "controller": "engineai_t800_walking_mnn",
                "opponent": "official_policy_approach_dummy",
                "combat_moves_enabled": False,
                "automatic_getup_enabled": True,
                "automatic_getup_profile": "engineai_t800_supine_to_stance_mnn",
                "recovering": self.recovering.copy(),
            }

    def frame(self) -> bytes:
        with self.lock:
            roots = np.vstack([binding.root_position(self.data) for binding in self.bindings])
            self.camera.lookat[:] = roots.mean(axis=0)
            self.camera.lookat[2] = max(0.8, self.camera.lookat[2])
            self.renderer.update_scene(self.data, camera=self.camera)
            return png_bytes(self.renderer.render())

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.renderer.close()


class Handler(BaseHTTPRequestHandler):
    evaluator: HumanEval | PolicyHumanEval

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode())
        elif path == "/health":
            self._send(200, "application/json", b'{"ok":true}')
        elif path == "/state":
            self._send(200, "application/json", json.dumps(self.evaluator.state()).encode())
        elif path == "/frame.png":
            self._send(200, "image/png", self.evaluator.frame())
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/input":
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            try:
                payload = json.loads(self.rfile.read(length))
                self.evaluator.update_action(payload)
            except (ValueError, TypeError, json.JSONDecodeError):
                self._send(400, "application/json", b'{"ok":false}')
                return
            self._send(200, "application/json", b'{"ok":true}')
        elif path == "/reset":
            self.evaluator.reset()
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def log_message(self, format: str, *args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--walking-policy", type=Path)
    parser.add_argument("--recovery-policy", type=Path)
    parser.add_argument("--recovery-trajectory", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    if args.walking_policy:
        if args.recovery_policy is None or args.recovery_trajectory is None:
            parser.error("--recovery-policy and --recovery-trajectory are required with --walking-policy")
        evaluator = PolicyHumanEval(
            args.model.resolve(),
            args.walking_policy.resolve(),
            args.recovery_policy.resolve(),
            args.recovery_trajectory.resolve(),
            args.log,
        )
    else:
        if args.library is None:
            parser.error("--library is required without --walking-policy")
        evaluator = HumanEval(args.library.resolve(), args.model.resolve(), args.log)
    Handler.evaluator = evaluator
    server = HTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ready": True, "host": args.host, "port": args.port}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        evaluator.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
