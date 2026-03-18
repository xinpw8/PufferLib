#!/usr/bin/env python3
"""test_flood_fill.py — Verify global coordinates and map placement integrity.

Checks:
  1. No two maps overlap (AABB intersection)
  2. All maps fit within the canvas bounds
  3. Warp destinations map to valid global coordinates
  4. All 425 maps are present in the global map JSON

Usage: python tools/test_flood_fill.py
"""

import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GLOBAL_MAP = REPO / "pufferlib" / "ocean" / "pfr_native" / "pfr_global_map.json"
EXTRACTED = Path("/tmp/pfr_extracted_maps.json")


def main():
    if not GLOBAL_MAP.exists():
        print(f"ERROR: {GLOBAL_MAP} not found. Run tools/gen_world_map.py first.")
        sys.exit(1)
    if not EXTRACTED.exists():
        print(f"ERROR: {EXTRACTED} not found. Run tools/dump_pfr_maps first.")
        sys.exit(1)

    gmap = json.loads(GLOBAL_MAP.read_text())
    raw = json.loads(EXTRACTED.read_text())
    raw_maps = {m["id"]: m for m in raw["maps"]}

    canvas_h, canvas_w = gmap["global_map_shape"]
    maps = gmap["maps"]

    failures = []
    total_checks = 0

    # 1. Check all 425 maps present
    total_checks += 1
    expected = len(raw_maps)
    actual = len(maps)
    if actual != expected:
        failures.append(f"Map count mismatch: expected {expected}, got {actual}")
        missing = set(str(k) for k in raw_maps.keys()) - set(maps.keys())
        if missing:
            failures.append(f"  Missing map IDs: {sorted(missing)[:20]}")
    print(f"Maps: {actual}/{expected}")

    # 2. All maps within canvas bounds
    for mid_str, m in maps.items():
        total_checks += 1
        gx, gy, w, h = m["gx"], m["gy"], m["width"], m["height"]
        if gx < 0 or gy < 0 or gx + w > canvas_w or gy + h > canvas_h:
            failures.append(f"Map {m['name']}({mid_str}) out of bounds: ({gx},{gy})+({w},{h}) > canvas({canvas_w},{canvas_h})")

    # 3. No overlap (AABB intersection)
    items = list(maps.items())
    overlap_count = 0
    for i in range(len(items)):
        mid_a, a = items[i]
        for j in range(i + 1, len(items)):
            mid_b, b = items[j]
            total_checks += 1
            if (a["gx"] < b["gx"] + b["width"] and a["gx"] + a["width"] > b["gx"] and
                    a["gy"] < b["gy"] + b["height"] and a["gy"] + a["height"] > b["gy"]):
                overlap_count += 1
                if overlap_count <= 5:
                    failures.append(f"Overlap: {a['name']}({mid_a}) vs {b['name']}({mid_b})")

    if overlap_count > 5:
        failures.append(f"  ... and {overlap_count - 5} more overlaps")
    print(f"Overlap check: {overlap_count} overlaps")

    # 4. Warp coordinate validity
    warp_checks = 0
    warp_fails = 0
    for mid_str, m in maps.items():
        mid = int(mid_str)
        if mid not in raw_maps:
            continue
        rm = raw_maps[mid]
        for warp in rm["warps"]:
            warp_checks += 1
            dest = warp["dest_map"]
            dest_str = str(dest)
            # Source warp global coordinate
            src_gx = m["gx"] + warp["x"]
            src_gy = m["gy"] + warp["y"]
            if src_gx < 0 or src_gx >= canvas_w or src_gy < 0 or src_gy >= canvas_h:
                warp_fails += 1
                if warp_fails <= 5:
                    failures.append(f"Warp source OOB: {m['name']}({mid_str}) warp ({warp['x']},{warp['y']}) -> global ({src_gx},{src_gy})")

            # Dest warp global coordinate
            if dest_str in maps and dest in raw_maps:
                dm = maps[dest_str]
                drm = raw_maps[dest]
                dwid = warp["dest_warp_id"]
                if dwid < len(drm["warps"]):
                    dw = drm["warps"][dwid]
                    dst_gx = dm["gx"] + dw["x"]
                    dst_gy = dm["gy"] + dw["y"]
                    if dst_gx < 0 or dst_gx >= canvas_w or dst_gy < 0 or dst_gy >= canvas_h:
                        warp_fails += 1
                        if warp_fails <= 5:
                            failures.append(f"Warp dest OOB: {dm['name']}({dest_str}) warp ({dw['x']},{dw['y']}) -> global ({dst_gx},{dst_gy})")

    print(f"Warp global coords: {warp_checks} checked, {warp_fails} OOB")

    # Summary
    print(f"\nFlood-Fill Accuracy Results")
    print(f"==========================")
    print(f"Total checks: {total_checks + warp_checks}")
    print(f"Failures:     {len(failures)}")

    if failures:
        print(f"\nFailures:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
