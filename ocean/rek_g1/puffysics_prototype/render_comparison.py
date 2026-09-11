"""Offscreen visualization of recorded candidate trajectories, without stepping.

Only mj_forward reconstructs drawable geometry. Render-only wall visibility,
colors and camera never modify the shared model file or either recorded run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(directory):
    report_path, trace_path = directory / "report.json", directory / "trace.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as archive:
        qpos = archive["qpos"].copy()
        elapsed = archive["elapsed_simulation_time"].copy()
    if qpos.ndim != 3 or qpos.shape[-1] != 72 or qpos.shape[0] < 1:
        raise ValueError(f"{directory}: expected recorded qpos [samples, arenas, 72]")
    return report, qpos, {"run_directory": str(directory.resolve()),
                          "report_sha256": digest(report_path), "trace_sha256": digest(trace_path)}, elapsed


def selected_indices(count, stride, tiles):
    if count < 1 or stride < 1 or tiles < 1:
        raise ValueError("positive sample count, stride and tile count required")
    video = list(range(0, count, stride))
    tile = sorted(set(np.linspace(0, count-1, min(tiles, count)).round().astype(int).tolist()))
    return video, tile


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def compose_panels(left, right, *, seconds, tick, role, arena, invalid, stopped_at=(None, None)):
    height, width = left.shape[:2]
    result = Image.new("RGB", (width*2, height+96), "#101820")
    result.paste(Image.fromarray(left), (0, 64))
    result.paste(Image.fromarray(right), (width, 64))
    draw = ImageDraw.Draw(result)
    draw.text((18, 10), "MuJoCo candidate", font=font(24), fill="#f4f6f8")
    draw.text((width+18, 10), "Puffysics prototype", font=font(24), fill="#f4f6f8")
    for side, stop in enumerate(stopped_at):
        actual_time = seconds if stop is None else min(seconds, stop)
        label = f"{role} | arena {arena} | t={actual_time:.3f} s | recorded tick {tick}"
        draw.text((side*width+18, 39), label, font=font(15), fill="#bed0df")
        if stop is not None and seconds > stop+1e-3:
            draw.text((side*width+18, 80), "SIMULATION HALTED: SOLVER FAILURE", font=font(20), fill="#ff7070")
    for side, failed in enumerate(invalid):
        if failed:
            draw.text((side*width+28, 110), "NONFINITE RECORDED STATE", font=font(22), fill="#ff7070")
    draw.line((width, 0, width, height+96), fill="#82909e", width=2)
    draw.text((18, height+71), "Playback of recorded CUDA simulation. Authentic REK parity is not established.",
              font=font(17), fill="#f4ce87")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arena", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--format", choices=("video", "tiles", "both", "gif"), default="both")
    parser.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="egl")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--distance-m", type=float, default=4.8)
    parser.add_argument("--elevation-deg", type=float, default=-40)
    parser.add_argument("--azimuth-deg", type=float, default=90)
    args = parser.parse_args()
    if min(args.stride, args.tiles, args.width, args.height) < 1 or args.width % 2 or args.height % 2:
        parser.error("positive stride/tiles and positive even image dimensions are required")
    if not np.isfinite([args.distance_m, args.elevation_deg, args.azimuth_deg]).all() or args.distance_m <= 0:
        parser.error("camera parameters must be finite and distance positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_path = Path(config["model"]).resolve(strict=True)
    if digest(model_path) != config["model_sha256"]:
        raise ValueError("shared model SHA-256 mismatch")
    reference_report, reference, reference_identity, reference_elapsed = load_trace(args.reference_run)
    candidate_report, candidate, candidate_identity, candidate_elapsed = load_trace(args.candidate_run)
    if reference_report["backend"] != "mujoco" or candidate_report["backend"] != "puffysics":
        raise ValueError("expected MuJoCo reference and Puffysics candidate run labels")
    for key in ("model_sha256", "assets_sha256", "fixed_reference_role", "reference_loop", "controller_period_seconds"):
        if reference_report[key] != candidate_report[key]:
            raise ValueError(f"recorded runs differ in {key}")
    if reference_report["model_sha256"] != config["model_sha256"]:
        raise ValueError("render model differs from recorded model")
    if not 0 <= args.arena < min(reference.shape[1], candidate.shape[1]):
        raise ValueError("selected arena is unavailable in one of the traces")
    stopped_at = []
    for report, elapsed in ((reference_report, reference_elapsed), (candidate_report, candidate_elapsed)):
        stats = report.get("native_stats", {}).get("per_arena", [])
        failed = bool(stats and (stats[args.arena][1] or stats[args.arena][2]))
        stopped_at.append(float(elapsed[args.arena]) if failed else None)
    count = min(reference.shape[0], candidate.shape[0])
    period = float(reference_report["controller_period_seconds"])
    if not np.isfinite(period) or period <= 0:
        raise ValueError("recorded controller period must be positive and finite")
    video_indices, tile_indices = selected_indices(count, args.stride, args.tiles)
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["MUJOCO_GL"] = args.gl
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 72:
        raise ValueError("render model QPOS width differs from recorded trace")
    hidden = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith(("arena_Collider_Wall_", "arena_Collider_Pillar_")):
            model.geom_rgba[geom, 3] = 0
            hidden.append(name)
        elif name.startswith("player__"):
            model.geom_rgba[geom] = [0.12, 0.42, 0.95, 1]
        elif name.startswith("opponent__"):
            model.geom_rgba[geom] = [1, 0.32, 0.06, 1]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.45]
    camera.distance = args.distance_m
    camera.azimuth = args.azimuth_deg
    camera.elevation = args.elevation_deg
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    writer, video_error = None, None
    video_path = args.out / "comparison.mp4"
    if args.format in ("video", "both"):
        try:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(video_path), fps=1/(period*args.stride),
                                       codec="libx264", quality=8, macro_block_size=1,
                                       pixelformat="yuv420p")
        except (ImportError, RuntimeError, ValueError) as exc:
            video_error = str(exc)
    gif_frames = []
    want_tiles = args.format in ("tiles", "both", "gif") or writer is None
    indices = sorted(set(video_indices if writer is not None or args.format == "gif" else []) | set(tile_indices if want_tiles else []))
    tiles, invalid_states, forward_calls, written_video_frames = {}, [], 0, 0
    try:
        for tick in indices:
            panels, invalid = [], []
            for label, trace in (("reference", reference), ("candidate", candidate)):
                qpos = trace[tick, args.arena]
                failed = not np.isfinite(qpos).all()
                invalid.append(failed)
                if failed:
                    invalid_states.append({"side": label, "tick": tick})
                    panels.append(np.zeros((args.height, args.width, 3), dtype=np.uint8))
                    continue
                data.qpos[:] = qpos
                data.qvel[:] = 0
                data.time = tick*period
                mujoco.mj_forward(model, data)
                forward_calls += 1
                renderer.update_scene(data, camera=camera)
                panels.append(renderer.render().copy())
            frame = compose_panels(*panels, seconds=tick*period, tick=tick,
                                   role=reference_report["fixed_reference_role"], arena=args.arena,
                                   invalid=invalid, stopped_at=stopped_at)
            if writer is not None and tick in video_indices:
                writer.append_data(np.asarray(frame))
                written_video_frames += 1
            if args.format == "gif" and tick in video_indices:
                gif_frames.append(frame.quantize(colors=192))
            if want_tiles and tick in tile_indices:
                tiles[tick] = frame
    finally:
        renderer.close()
        if writer is not None:
            writer.close()
    artifacts = []
    if written_video_frames:
        artifacts.append({"kind": "video", "path": str(video_path.resolve()), "sha256": digest(video_path)})
    if gif_frames:
        gif_path = args.out / "comparison.gif"
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                           duration=round(period*args.stride*1000), loop=0, disposal=2)
        artifacts.append({"kind": "animated_gif", "path": str(gif_path.resolve()), "sha256": digest(gif_path)})
    if tiles:
        tile_height = args.height+96
        sheet = Image.new("RGB", (args.width*2, tile_height*len(tiles)), "#101820")
        for row, tick in enumerate(sorted(tiles)):
            sheet.paste(tiles[tick], (0, row*tile_height))
        sheet_path = args.out / "contact_sheet.png"
        sheet.save(sheet_path)
        artifacts.append({"kind": "static_frames", "path": str(sheet_path.resolve()), "sha256": digest(sheet_path)})
    if digest(model_path) != config["model_sha256"]:
        raise RuntimeError("source model digest changed during rendering")
    report = {
        "schema": "rek.puffysics_recorded_state_render.v1", "artifacts": artifacts,
        "reference": reference_identity, "candidate": candidate_identity,
        "shared_model_sha256": config["model_sha256"], "selected_arena": args.arena,
        "available_common_samples": count, "rendered_ticks": indices,
        "video_frames": written_video_frames, "video_fps": 1/(period*args.stride),
        "gif_frames": len(gif_frames), "gif_frame_duration_ms": round(period*args.stride*1000),
        "video_error": video_error, "tiled_ticks": sorted(tiles), "invalid_states": invalid_states,
        "solver_halt_times_seconds": stopped_at,
        "mj_forward_calls_for_geometry": forward_calls, "mj_step_calls": 0,
        "render_only_hidden_geometries": hidden,
        "render_only_fighter_palette": "player blue, opponent orange",
        "camera": {"lookat_m": [0, 0, 0.45], "distance_m": args.distance_m,
                   "elevation_deg": args.elevation_deg, "azimuth_deg": args.azimuth_deg},
        "authentic_rek_parity_established": False, "render_does_not_run_controller_or_physics_steps": True,
    }
    with (args.out / "render_report.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
