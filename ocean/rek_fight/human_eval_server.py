#!/usr/bin/env python3
import argparse
import ctypes
import json
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
  <p class="notice">Provisional simulator. Blue is you. Orange is a deterministic move dummy, not recovered Bot 1.</p>
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
    <kbd>1</kbd>–<kbd>6</kbd> moves.
    Click this page once before using the keyboard.
  </p>
  <button id="reset" type="button">Reset round</button><span id="connection">connecting</span>
</main>
<script>
(() => {
  const held = new Set();
  const relevant = new Set(['KeyW','KeyS','KeyA','KeyD','KeyQ','KeyE','Digit1','Digit2','Digit3','Digit4','Digit5','Digit6']);
  let queuedMove = 0;
  let inputSequence = 0;
  const axis = (negative, positive) => held.has(negative) === held.has(positive) ? 1 : held.has(positive) ? 2 : 0;
  async function sendInput() {
    const payload = {
      sequence: ++inputSequence,
      forward: axis('KeyS', 'KeyW'),
      strafe: axis('KeyA', 'KeyD'),
      yaw: axis('KeyQ', 'KeyE'),
      move: queuedMove
    };
    queuedMove = 0;
    try {
      await fetch('/input', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
    } catch (_) {}
  }
  addEventListener('keydown', event => {
    if (!relevant.has(event.code)) return;
    event.preventDefault();
    if (event.code.startsWith('Digit') && !event.repeat) queuedMove = Number(event.code.slice(5));
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
    held.clear(); queuedMove = 0;
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
      document.getElementById('connection').textContent = 'live at 50 Hz';
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
        self.camera.distance = 6.5
        self.camera.azimuth = 90.0
        self.camera.elevation = -68.0
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
                "opponent": "deterministic_move_dummy",
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


class Handler(BaseHTTPRequestHandler):
    evaluator: HumanEval

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
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

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
