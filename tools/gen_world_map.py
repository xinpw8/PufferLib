#!/usr/bin/env python3
"""gen_world_map.py — Generate pixel-perfect global world map for pfr_native.

Algorithm:
  1. BFS outdoor maps (Kanto, Sevii) via map connections → deterministic layout
  2. Place indoor maps near their warp entrances, spiral on collision
  3. Render world_map.png, output pfr_global_map.json and pfr_heatmap_lut.h

Usage:
  python tools/gen_world_map.py
"""

import json, os, sys, heapq
from collections import defaultdict, deque
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
PFR_NATIVE_DIR = REPO / "pufferlib" / "ocean" / "pfr_native"
RESOURCE_DIR = REPO / "pufferlib" / "resources" / "pfr_native"
EXTRACTED = Path("/tmp/pfr_extracted_maps.json")

# Connection directions
CONN_NORTH, CONN_SOUTH, CONN_WEST, CONN_EAST = 0, 1, 2, 3

PAD = 2  # tiles between maps to avoid visual merging


def load_maps():
    """Load extracted map data from the C dumper tool."""
    if not EXTRACTED.exists():
        print(f"ERROR: {EXTRACTED} not found. Run: ./tools/dump_pfr_maps > /tmp/pfr_extracted_maps.json")
        sys.exit(1)
    data = json.loads(EXTRACTED.read_text())
    maps = {}
    for m in data["maps"]:
        maps[m["id"]] = m
    return maps


def bfs_outdoor(maps, start_id):
    """BFS through connections to place outdoor maps.
    Returns dict of map_id -> (gx, gy) in tile coordinates.
    """
    placed = {}
    if start_id not in maps:
        return placed
    start = maps[start_id]
    placed[start_id] = (0, 0)
    queue = deque([start_id])

    while queue:
        mid = queue.popleft()
        m = maps[mid]
        gx, gy = placed[mid]

        for conn in m["connections"]:
            dest = conn["dest_map"]
            if dest in placed or dest not in maps:
                continue
            dm = maps[dest]
            d = conn["direction"]
            off = conn["offset"]

            if d == CONN_NORTH:
                dx = gx + off
                dy = gy - dm["height"]
            elif d == CONN_SOUTH:
                dx = gx + off
                dy = gy + m["height"]
            elif d == CONN_WEST:
                dx = gx - dm["width"]
                dy = gy + off
            elif d == CONN_EAST:
                dx = gx + m["width"]
                dy = gy + off
            else:
                continue

            placed[dest] = (dx, dy)
            queue.append(dest)

    return placed


class SpatialGrid:
    """Fast overlap detection using a grid of occupied cells."""

    def __init__(self):
        self.occupied = set()  # set of (x, y) occupied tiles

    def mark(self, gx, gy, w, h):
        for y in range(gy, gy + h):
            for x in range(gx, gx + w):
                self.occupied.add((x, y))

    def overlaps(self, gx, gy, w, h):
        for y in range(gy, gy + h):
            for x in range(gx, gx + w):
                if (x, y) in self.occupied:
                    return True
        return False


def spiral_place(grid, ideal_x, ideal_y, w, h, max_radius=500):
    """Find nearest non-overlapping position to (ideal_x, ideal_y) using spiral search."""
    # Priority queue: (distance, x, y)
    pq = [(0, ideal_x, ideal_y)]
    visited = set()

    while pq:
        dist, cx, cy = heapq.heappop(pq)
        key = (cx, cy)
        if key in visited:
            continue
        visited.add(key)

        if not grid.overlaps(cx, cy, w, h):
            return cx, cy

        # Expand to 4 neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            nkey = (nx, ny)
            if nkey not in visited:
                ndist = abs(nx - ideal_x) + abs(ny - ideal_y)
                if ndist <= max_radius:
                    heapq.heappush(pq, (ndist, nx, ny))

    # Fallback: place far away
    return ideal_x + max_radius, ideal_y


def place_indoor_maps(maps, placed, grid):
    """Place all unplaced (indoor) maps near their warp entrances."""
    # Build reverse warp index: for each map, which placed maps have warps TO it?
    # Also build: for each unplaced map, which placed maps does it warp TO?
    unplaced = [mid for mid in maps if mid not in placed]

    # Sort by: maps that warp to already-placed maps first (direct children)
    # Then iterate until all placed
    max_iterations = 20
    for iteration in range(max_iterations):
        newly_placed = []
        still_unplaced = []

        for mid in unplaced:
            m = maps[mid]
            best_pos = None
            best_dist = float("inf")

            # Check this map's warps — does any lead to an already-placed map?
            for warp in m["warps"]:
                dest = warp["dest_map"]
                if dest not in placed or dest not in maps:
                    continue
                dm = maps[dest]
                dgx, dgy = placed[dest]

                # Find the destination warp tile on the placed map
                if warp["dest_warp_id"] < len(dm["warps"]):
                    dest_warp = dm["warps"][warp["dest_warp_id"]]
                    # Ideal: align this map's warp tile with the dest warp tile
                    ideal_x = dgx + dest_warp["x"] - warp["x"]
                    ideal_y = dgy + dest_warp["y"] - warp["y"]
                else:
                    # Fallback: place near the destination map center
                    ideal_x = dgx + dm["width"] // 2 - m["width"] // 2
                    ideal_y = dgy + dm["height"] + PAD

                dist = abs(ideal_x - dgx) + abs(ideal_y - dgy)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (ideal_x, ideal_y)

            # Also check: does any already-placed map have warps TO this map?
            if best_pos is None:
                for pid, (pgx, pgy) in placed.items():
                    pm = maps[pid]
                    for warp in pm["warps"]:
                        if warp["dest_map"] == mid:
                            # Place below the parent map's warp
                            ideal_x = pgx + warp["x"] - m["width"] // 2
                            ideal_y = pgy + pm["height"] + PAD
                            dist = abs(ideal_x - pgx) + abs(ideal_y - pgy)
                            if dist < best_dist:
                                best_dist = dist
                                best_pos = (ideal_x, ideal_y)

            if best_pos is not None:
                w, h = m["width"] + PAD, m["height"] + PAD
                fx, fy = spiral_place(grid, best_pos[0], best_pos[1], w, h)
                placed[mid] = (fx, fy)
                grid.mark(fx, fy, w, h)
                newly_placed.append(mid)
            else:
                still_unplaced.append(mid)

        unplaced = still_unplaced
        if not newly_placed:
            break  # No progress

    # Any remaining unplaced maps: stack them in a fallback region
    if unplaced:
        # Find bottom-right corner of all placed maps
        max_y = max(gy + maps[mid]["height"] for mid, (gx, gy) in placed.items()) if placed else 0
        fallback_x = 0
        fallback_y = max_y + 20

        for mid in unplaced:
            m = maps[mid]
            w, h = m["width"] + PAD, m["height"] + PAD
            fx, fy = spiral_place(grid, fallback_x, fallback_y, w, h)
            placed[mid] = (fx, fy)
            grid.mark(fx, fy, w, h)
            fallback_x = fx + w + PAD

        print(f"  Placed {len(unplaced)} orphan maps in fallback region")


def normalize_positions(placed, maps, border=20):
    """Shift all positions so min is (border, border)."""
    if not placed:
        return placed, 0, 0
    min_x = min(gx for gx, gy in placed.values())
    min_y = min(gy for gx, gy in placed.values())
    shifted = {}
    for mid, (gx, gy) in placed.items():
        shifted[mid] = (gx - min_x + border, gy - min_y + border)
    max_x = max(gx + maps[mid]["width"] for mid, (gx, gy) in shifted.items())
    max_y = max(gy + maps[mid]["height"] for mid, (gx, gy) in shifted.items())
    canvas_w = max_x + border
    canvas_h = max_y + border
    return shifted, canvas_w, canvas_h


def validate_no_overlap(placed, maps):
    """Verify all map rectangles are non-overlapping."""
    rects = []
    for mid, (gx, gy) in placed.items():
        m = maps[mid]
        rects.append((mid, gx, gy, m["width"], m["height"]))

    overlaps = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a_id, ax, ay, aw, ah = rects[i]
            b_id, bx, by, bw, bh = rects[j]
            if ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by:
                overlaps += 1
                if overlaps <= 5:
                    print(f"  OVERLAP: {maps[a_id]['name']}({a_id}) @ ({ax},{ay},{aw}x{ah}) "
                          f"vs {maps[b_id]['name']}({b_id}) @ ({bx},{by},{bw}x{bh})")
    return overlaps


def render_world_map(placed, maps, canvas_w, canvas_h, output_path):
    """Render the world map as a PNG."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("WARNING: Pillow not installed, skipping PNG render. pip install Pillow")
        return False

    # Color scheme by map type
    def map_color(m):
        name = m["name"].lower()
        has_conns = len(m["connections"]) > 0

        # Outdoor towns/cities
        if has_conns:
            if "route" in name or "seaRoute" in name.replace(" ", ""):
                return (180, 210, 160)  # pale green for routes
            if "island" in name:
                return (200, 180, 140)  # sandy for islands
            return (200, 200, 220)  # light blue-gray for towns

        # Indoor categories
        if "pokemoncenter" in name.lower().replace("_", ""):
            return (255, 180, 180)  # pink
        if "mart" in name.lower() or "shop" in name.lower():
            return (180, 200, 255)  # light blue
        if "gym" in name.lower():
            return (255, 220, 140)  # gold
        if "cave" in name.lower() or "tunnel" in name.lower() or "mtmoon" in name.lower():
            return (140, 130, 120)  # brown-gray
        if "ssanne" in name.lower():
            return (160, 200, 240)  # ocean blue
        if "house" in name.lower() or "home" in name.lower():
            return (220, 200, 180)  # beige
        if "lab" in name.lower():
            return (200, 220, 200)  # light green
        return (190, 190, 190)  # default gray

    # Scale: 1 tile = 1 pixel for compact output
    scale = 1
    img = Image.new("RGB", (canvas_w * scale, canvas_h * scale), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    for mid, (gx, gy) in placed.items():
        m = maps[mid]
        x0 = gx * scale
        y0 = gy * scale
        x1 = (gx + m["width"]) * scale - 1
        y1 = (gy + m["height"]) * scale - 1

        fill = map_color(m)
        draw.rectangle([x0, y0, x1, y1], fill=fill)
        draw.rectangle([x0, y0, x1, y1], outline=(60, 60, 60))

        # Draw warp markers
        for warp in m["warps"]:
            if not warp["supported"]:
                continue
            wx = (gx + warp["x"]) * scale
            wy = (gy + warp["y"]) * scale
            if 0 <= wx < img.width and 0 <= wy < img.height:
                draw.point((wx, wy), fill=(255, 80, 80))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG", optimize=True)
    print(f"  World map: {output_path} ({canvas_w}x{canvas_h} px, {output_path.stat().st_size // 1024} KB)")
    return True


def write_global_map_json(placed, maps, canvas_w, canvas_h, output_path):
    """Write pfr_global_map.json with unique coordinates for all maps."""
    pad = 20  # border pad for heatmap
    out = {
        "global_map_shape": [canvas_h, canvas_w],
        "padded_shape": [canvas_h + 2 * pad, canvas_w + 2 * pad],
        "pad": pad,
        "maps": {}
    }
    for mid in sorted(placed.keys()):
        m = maps[mid]
        gx, gy = placed[mid]
        out["maps"][str(mid)] = {
            "name": m["name"],
            "gx": gx,
            "gy": gy,
            "width": m["width"],
            "height": m["height"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  JSON: {output_path}")


def write_heatmap_lut(placed, maps, canvas_w, canvas_h, output_path):
    """Write pfr_heatmap_lut.h — C lookup table for heatmap coordinates."""
    pad = 20
    padded_w = canvas_w + 2 * pad
    padded_h = canvas_h + 2 * pad

    max_map_id = max(placed.keys())
    total_entries = max_map_id + 1

    lines = []
    lines.append("/* Auto-generated by tools/gen_world_map.py — do not edit */")
    lines.append(f"#define PFR_HEATMAP_H {padded_h}")
    lines.append(f"#define PFR_HEATMAP_W {padded_w}")
    lines.append(f"#define PFR_HEATMAP_SIZE ({padded_h} * {padded_w})")
    lines.append(f"#define PFR_HEATMAP_PAD {pad}")
    lines.append(f"#define PFR_HEATMAP_MAX_MAP_ID {max_map_id}")
    lines.append("")
    lines.append("typedef struct { int16_t gx; int16_t gy; } PfrMapOffset;")
    lines.append("")
    lines.append(f"static const PfrMapOffset pfr_map_offsets[{total_entries}] = {{")

    for i in range(total_entries):
        if i in placed:
            gx, gy = placed[i]
            # Add pad offset so heatmap coordinates include border
            lines.append(f"    [{i}] = {{ .gx = {gx + pad}, .gy = {gy + pad} }},")
        else:
            lines.append(f"    [{i}] = {{ .gx = -1, .gy = -1 }},")

    lines.append("};")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  LUT: {output_path}")


def main():
    print("Loading map data...")
    maps = load_maps()
    print(f"  {len(maps)} maps loaded")

    # Identify outdoor maps (those with connections)
    outdoor = {mid for mid, m in maps.items() if m["connections"]}
    print(f"  {len(outdoor)} outdoor maps with connections")

    # Phase 1: BFS outdoor Kanto maps from Pallet Town (190)
    print("\nPhase 1: BFS outdoor Kanto from Pallet Town...")
    kanto_placed = bfs_outdoor(maps, 190)
    print(f"  Placed {len(kanto_placed)} Kanto outdoor maps")

    # Phase 2: BFS outdoor Sevii Islands (each island is a separate connected component)
    sevii_starts = [202, 203, 204, 205, 206, 207, 208]  # Islands 1-7
    print("\nPhase 2: BFS outdoor Sevii Islands...")

    # Kanto bounds for positioning Sevii to the right
    kanto_max_x = max(gx + maps[mid]["width"] for mid, (gx, gy) in kanto_placed.items()) if kanto_placed else 0
    kanto_min_y = min(gy for gx, gy in kanto_placed.values()) if kanto_placed else 0

    # BFS each island group independently, then stack them vertically
    sevii_groups = []
    placed_set = set(kanto_placed.keys())
    for sid in sevii_starts:
        if sid in placed_set or sid not in maps:
            continue
        group = bfs_outdoor(maps, sid)
        # Remove any already-placed maps
        group = {mid: pos for mid, pos in group.items() if mid not in placed_set}
        if group:
            sevii_groups.append(group)
            placed_set.update(group.keys())

    # Also pick up any remaining outdoor maps not yet placed
    for mid in sorted(outdoor - placed_set):
        if mid in maps:
            group = bfs_outdoor(maps, mid)
            group = {m: p for m, p in group.items() if m not in placed_set}
            if group:
                sevii_groups.append(group)
                placed_set.update(group.keys())

    # Stack Sevii groups vertically to the right of Kanto
    sevii_placed = {}
    sevii_cursor_y = kanto_min_y
    for group in sevii_groups:
        # Normalize group to (0, 0)
        gmin_x = min(gx for gx, gy in group.values())
        gmin_y = min(gy for gx, gy in group.values())
        gmax_y = max(gy + maps[mid]["height"] for mid, (gx, gy) in group.items())

        for mid, (gx, gy) in group.items():
            sevii_placed[mid] = (
                gx - gmin_x + kanto_max_x + 40,
                gy - gmin_y + sevii_cursor_y
            )
        sevii_cursor_y += (gmax_y - gmin_y) + 20

    print(f"  Placed {len(sevii_placed)} Sevii/disconnected outdoor maps in {len(sevii_groups)} groups")

    # Merge outdoor placements
    placed = {}
    placed.update(kanto_placed)
    placed.update(sevii_placed)

    # Handle duplicate outdoor maps (e.g., SaffronCity vs SaffronCity_Connection)
    # that have identical connections but only one was reached by BFS
    for mid in sorted(outdoor - set(placed.keys())):
        m = maps[mid]
        if not m["connections"]:
            continue
        # Find a placed map with the same connection targets
        dest_set = frozenset(c["dest_map"] for c in m["connections"])
        for pid, (pgx, pgy) in list(placed.items()):
            pm = maps[pid]
            if not pm["connections"]:
                continue
            pdest_set = frozenset(c["dest_map"] for c in pm["connections"])
            if dest_set == pdest_set and mid != pid:
                # Place adjacent (below) the duplicate
                placed[mid] = (pgx, pgy + pm["height"] + PAD)
                print(f"  Placed duplicate {m['name']}({mid}) near {pm['name']}({pid})")
                break

    print(f"\n  Total outdoor maps placed: {len(placed)}")

    # Phase 3: Place indoor maps near their warp entrances
    print("\nPhase 3: Placing indoor maps via warp adjacency...")
    grid = SpatialGrid()
    for mid, (gx, gy) in placed.items():
        m = maps[mid]
        grid.mark(gx, gy, m["width"] + PAD, m["height"] + PAD)

    place_indoor_maps(maps, placed, grid)
    print(f"  Total maps placed: {len(placed)}/{len(maps)}")

    # Normalize
    print("\nNormalizing positions...")
    placed, canvas_w, canvas_h = normalize_positions(placed, maps, border=20)

    # Validate
    print("\nValidating non-overlap...")
    overlap_count = validate_no_overlap(placed, maps)
    if overlap_count == 0:
        print("  PASS: All 0 overlaps")
    else:
        print(f"  FAIL: {overlap_count} overlaps detected")

    # Output
    print("\nGenerating outputs...")
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

    write_global_map_json(placed, maps, canvas_w, canvas_h,
                          PFR_NATIVE_DIR / "pfr_global_map.json")

    write_heatmap_lut(placed, maps, canvas_w, canvas_h,
                      PFR_NATIVE_DIR / "pfr_heatmap_lut.h")

    render_world_map(placed, maps, canvas_w, canvas_h,
                     RESOURCE_DIR / "world_map.png")

    print(f"\nDone! {len(placed)} maps placed on {canvas_w}x{canvas_h} canvas.")


if __name__ == "__main__":
    main()
