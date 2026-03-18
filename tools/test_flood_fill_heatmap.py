#!/usr/bin/env python3
"""test_flood_fill_heatmap.py — Flood-fill every map until 100% tile coverage.

For each map: spawn an agent at every tile, walk randomly until all walkable
tiles have been visited at least once. The C heatmap records every step.
Renders the result overlaid on the world map with magenta borders on top.

Usage: python tools/test_flood_fill_heatmap.py [--max-steps-per-tile 200]
"""

import json, re, sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
RESOURCE_DIR = REPO / "pufferlib" / "resources" / "pfr_native"
PFR_NATIVE_DIR = REPO / "pufferlib" / "ocean" / "pfr_native"
GLOBAL_MAP = PFR_NATIVE_DIR / "pfr_global_map.json"
LUT_H = PFR_NATIVE_DIR / "pfr_heatmap_lut.h"
WORLD_MAP_PNG = RESOURCE_DIR / "world_map.png"
OUTPUT = RESOURCE_DIR / "flood_fill_diagnostic.png"


def parse_lut(path):
    text = path.read_text()
    heatmap_h = int(re.search(r"#define PFR_HEATMAP_H (\d+)", text).group(1))
    heatmap_w = int(re.search(r"#define PFR_HEATMAP_W (\d+)", text).group(1))
    pad = int(re.search(r"#define PFR_HEATMAP_PAD (\d+)", text).group(1))
    offsets = {}
    for m in re.finditer(r"\[(\d+)\]\s*=\s*\{\s*\.gx\s*=\s*(-?\d+),\s*\.gy\s*=\s*(-?\d+)\s*\}", text):
        offsets[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return offsets, heatmap_h, heatmap_w, pad


def flood_fill_map(binding, handle, s_obs, s_act, map_id, w, h, max_steps_per_tile):
    """Walk an agent around a single map until every walkable tile is visited.
    Returns set of (local_x, local_y) walkable tiles discovered."""
    visited = set()

    # Phase 1: discover all walkable tiles by spawning at every grid cell
    spawn_tiles = []
    for ty in range(h):
        for tx in range(w):
            ok = binding.reset_to_map(handle, map_id, tx, ty)
            if ok != 0:
                continue
            s_act[0] = 0
            binding.env_step(handle)
            o = s_obs
            cm = int(o[4]) | (int(o[5]) << 8)
            cx = int(o[0]) | (int(o[1]) << 8)
            cy = int(o[2]) | (int(o[3]) << 8)
            if cm == map_id:
                visited.add((cx, cy))
                spawn_tiles.append((tx, ty))

    if not spawn_tiles:
        return visited

    total_walkable = len(visited)

    # Phase 2: random-walk from unvisited-adjacent tiles until full coverage
    # We keep a working set of tiles that still have unvisited neighbors
    stale_count = 0
    steps = 0
    max_total = total_walkable * max_steps_per_tile

    while steps < max_total:
        # Pick a random spawn point
        sx, sy = spawn_tiles[np.random.randint(len(spawn_tiles))]
        binding.reset_to_map(handle, map_id, sx, sy)

        for _ in range(max_steps_per_tile):
            s_act[0] = float(np.random.randint(1, 5))  # up/down/left/right
            binding.env_step(handle)
            steps += 1
            o = s_obs
            cm = int(o[4]) | (int(o[5]) << 8)
            if cm != map_id:
                break
            cx = int(o[0]) | (int(o[1]) << 8)
            cy = int(o[2]) | (int(o[3]) << 8)
            prev_count = len(visited)
            visited.add((cx, cy))
            if len(visited) > prev_count:
                stale_count = 0
            else:
                stale_count += 1

        # Check convergence: if we've gone a long time with no new tiles, we're done
        if stale_count > total_walkable * 10:
            break

    return visited


def render(heatmap, gmap, lut_offsets, pad, corners, output_path):
    heatmap_h, heatmap_w = heatmap.shape

    if WORLD_MAP_PNG.exists():
        bg = np.array(Image.open(str(WORLD_MAP_PNG)).convert("RGB"), dtype=np.uint8)
    else:
        bg = np.full((100, 100, 3), 30, dtype=np.uint8)

    canvas = np.full((heatmap_h, heatmap_w, 3), 30, dtype=np.uint8)
    bh, bw = bg.shape[:2]
    y_end = min(pad + bh, heatmap_h)
    x_end = min(pad + bw, heatmap_w)
    canvas[pad:y_end, pad:x_end] = bg[:y_end - pad, :x_end - pad]

    # Heatmap overlay
    nz = heatmap > 0
    if np.any(nz):
        log_c = np.log1p(heatmap)
        log_mx = np.max(log_c)
        if log_mx > 0:
            scaled = log_c / log_mx
        else:
            scaled = log_c
        nz_f = nz.astype(np.float32)
        h = 2.0 * (1.0 - scaled) / 3.0
        from pufferlib.ocean.pfr_native.pfr_heatmap_callback import _hsv_to_rgb
        heat_rgb = _hsv_to_rgb(h, nz_f, nz_f)
        heat_u8 = (255 * heat_rgb).astype(np.uint8)

        alpha = 0.6 * nz_f
        a3 = alpha[:, :, np.newaxis]
        canvas = (canvas.astype(np.float32) * (1 - a3) +
                  heat_u8.astype(np.float32) * a3).clip(0, 255).astype(np.uint8)

    # Corner markers: cyan
    for gy, gx in corners:
        if 0 <= gy < heatmap_h and 0 <= gx < heatmap_w:
            canvas[gy, gx] = [0, 255, 255]

    # Magenta borders on top
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for mid_str, m in gmap["maps"].items():
        mid = int(mid_str)
        if mid not in lut_offsets:
            continue
        lgx, lgy = lut_offsets[mid]
        if lgx < 0:
            continue
        draw.rectangle([lgx, lgy, lgx + m["width"] - 1, lgy + m["height"] - 1],
                        outline=(255, 0, 255))

    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    img.save(str(output_path), "PNG", optimize=True)
    return img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps-per-tile", type=int, default=200)
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    args = parser.parse_args()

    gmap = json.loads(GLOBAL_MAP.read_text())
    lut_offsets, heatmap_h, heatmap_w, pad = parse_lut(LUT_H)

    from pufferlib.ocean.pfr_native import binding

    # Dummy vec env to ensure heatmap is allocated
    d_obs = np.zeros((1, 129), dtype=np.uint8)
    d_act = np.zeros((1, 1), dtype=np.float32)
    d_rew = np.zeros(1, dtype=np.float32)
    d_ter = np.zeros(1, dtype=np.float32)
    d_tru = np.zeros(1, dtype=np.float32)
    vec = binding.vec_init(d_obs, d_act, d_rew, d_ter, d_tru, 1, 0)
    binding.vec_reset(vec, 0)

    heatmap = binding.get_heatmap()
    heatmap[:] = 0

    # Single env for targeted spawning
    s_obs = np.zeros(129, dtype=np.uint8)
    s_act = np.zeros(1, dtype=np.float32)
    s_rew = np.zeros(1, dtype=np.float32)
    s_ter = np.zeros(1, dtype=np.float32)
    s_tru = np.zeros(1, dtype=np.float32)
    handle = binding.env_init(s_obs, s_act, s_rew, s_ter, s_tru, 42)

    num_maps = binding.get_map_count()
    print(f"Flood-filling {num_maps} maps (heatmap {heatmap_w}x{heatmap_h}, pad={pad})")

    corners = []  # list of (gy, gx) for cyan markers
    total_walkable = 0
    maps_done = 0

    for map_id in range(num_maps):
        name, w, h, wc = binding.get_map_info(map_id)
        if map_id not in lut_offsets or lut_offsets[map_id][0] < 0:
            continue
        lgx, lgy = lut_offsets[map_id]

        walked = flood_fill_map(binding, handle, s_obs, s_act, map_id, w, h,
                                args.max_steps_per_tile)
        maps_done += 1
        total_walkable += len(walked)

        # Corner markers from walkable tiles
        if walked:
            tl = min(walked, key=lambda t: (t[1], t[0]))
            tr = min(walked, key=lambda t: (t[1], -t[0]))
            bl = min(walked, key=lambda t: (-t[1], t[0]))
            br = min(walked, key=lambda t: (-t[1], -t[0]))
            for cx, cy in [tl, tr, bl, br]:
                corners.append((cy + lgy, cx + lgx))

        if (map_id + 1) % 50 == 0 or map_id == num_maps - 1:
            print(f"  {maps_done}/{num_maps} maps, {total_walkable} walkable tiles, "
                  f"heatmap non-zero={np.count_nonzero(heatmap)}")

    binding.env_close(handle)
    binding.vec_close(vec)

    print(f"\nMaps flood-filled: {maps_done}")
    print(f"Total walkable tiles: {total_walkable}")
    print(f"Heatmap pixels with visits: {np.count_nonzero(heatmap)}")
    print(f"Corner markers: {len(corners)}")

    out = Path(args.output)
    img = render(heatmap, gmap, lut_offsets, pad, corners, out)
    print(f"Saved: {out} ({out.stat().st_size // 1024} KB, {img.width}x{img.height})")


if __name__ == "__main__":
    main()
