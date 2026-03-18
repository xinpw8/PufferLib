#!/usr/bin/env python3
"""test_warps.py — Verify all warps across pfr_native maps.

Checks:
  1. Every warp's dest_map is a valid map ID
  2. Every warp's dest_warp_id is a valid index on the dest map
  3. Destination warp coordinates are within dest map bounds
  4. Bidirectional: dest map has a warp back to source

Usage: python tools/test_warps.py
"""

import json, sys
from pathlib import Path

EXTRACTED = Path("/tmp/pfr_extracted_maps.json")


def main():
    if not EXTRACTED.exists():
        print(f"ERROR: {EXTRACTED} not found")
        sys.exit(1)

    data = json.loads(EXTRACTED.read_text())
    maps = {m["id"]: m for m in data["maps"]}

    MAP_INVALID = 65535  # PFR_NATIVE_MAP_INVALID — dynamic dest (elevators, battle facilities)

    # Maps known to have one-way warps (elevators, fall-through holes)
    elevator_names = {"elevator", "dotted_hole"}

    total = 0
    passed = 0
    skipped = 0
    failures = []   # structural errors (broken data)
    warnings = []   # one-way warps (legitimate game mechanics)

    for mid, m in sorted(maps.items()):
        for wi, warp in enumerate(m["warps"]):
            total += 1
            errors = []
            dest = warp["dest_map"]
            dwid = warp["dest_warp_id"]

            # Skip dynamic warps (dest set at runtime by scripts)
            if dest == MAP_INVALID:
                skipped += 1
                passed += 1
                continue

            # 1. Valid dest_map
            if dest not in maps:
                errors.append(f"dest_map {dest} not found")
            else:
                dm = maps[dest]

                # 2. Valid dest_warp_id
                if dwid >= len(dm["warps"]):
                    errors.append(f"dest_warp_id {dwid} out of range (dest has {len(dm['warps'])} warps)")
                else:
                    dw = dm["warps"][dwid]
                    # 3. Dest warp in bounds
                    if dw["x"] < 0 or dw["x"] >= dm["width"] or dw["y"] < 0 or dw["y"] >= dm["height"]:
                        errors.append(f"dest warp ({dw['x']},{dw['y']}) out of bounds for {dm['name']} ({dm['width']}x{dm['height']})")

                # 4. Bidirectional check (warning only — one-way warps are legitimate)
                has_return = any(w["dest_map"] == mid for w in dm["warps"])
                if not has_return:
                    warnings.append(f"  {m['name']}({mid}) warp[{wi}] -> {dm['name']}({dest}): one-way (no return)")

            if errors:
                for e in errors:
                    failures.append(f"  {m['name']}({mid}) warp[{wi}] ({warp['x']},{warp['y']}) -> map{dest}[{dwid}]: {e}")
            else:
                passed += 1

    print(f"\nWarp Verification Results")
    print(f"========================")
    print(f"Total warps: {total}")
    print(f"Passed:      {passed}")
    print(f"Skipped:     {skipped} (dynamic destinations)")
    print(f"Warnings:    {len(warnings)} (one-way warps)")
    print(f"Failed:      {len(failures)}")

    if failures:
        print(f"\nFailures:")
        for f in failures[:50]:
            print(f)
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        sys.exit(1)
    else:
        print("\nAll warps verified OK!")
        sys.exit(0)


if __name__ == "__main__":
    main()
