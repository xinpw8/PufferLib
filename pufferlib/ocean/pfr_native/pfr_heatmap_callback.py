"""
pfr_heatmap_callback.py — Exploration heatmap overlay on world map.

Reads the shared heatmap buffer accumulated in C (zero-copy via _C.get_heatmap)
and composites a visit heatmap over the static world_map_textured.png for wandb logging.
No per-step Python overhead — render only on epoch boundaries.
"""

import numpy as np
from pathlib import Path

_WORLD_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "pfr_native" / "world_map_textured.png"
_cached_bg = None


def _load_world_map_bg(target_h, target_w):
    global _cached_bg
    if _cached_bg is not None:
        return _cached_bg
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        if _WORLD_MAP_PATH.exists():
            img = Image.open(str(_WORLD_MAP_PATH)).convert("RGB")
            img = img.resize((target_w, target_h), Image.LANCZOS)
            _cached_bg = np.array(img, dtype=np.uint8)
        else:
            _cached_bg = None
    except ImportError:
        _cached_bg = None
    return _cached_bg


class PfrHeatmapCallback:
    def __init__(self, interval=10):
        self.interval = max(1, int(interval))
        self._calls = 0

    def should_render(self):
        self._calls += 1
        return self._calls % self.interval == 0

    def render(self, heatmap_np):
        mx = np.max(heatmap_np)
        if mx == 0:
            return None

        hh, hw = heatmap_np.shape
        bg = _load_world_map_bg(hh, hw)

        log_c = np.log1p(heatmap_np)
        log_mx = np.max(log_c)
        scaled = log_c / log_mx
        nz = (heatmap_np > 0).astype(np.float32)

        # HSV heatmap: blue (low) -> red (high)
        h = 2.0 * (1.0 - scaled) / 3.0
        heat_rgb = _hsv_to_rgb(h, nz, nz)
        heat_u8 = (255 * heat_rgb).astype(np.uint8)

        if bg is not None:
            alpha = (0.7 * nz + 0.3 * scaled * nz)
            alpha_3 = alpha[:, :, np.newaxis]

            bg_f = bg.astype(np.float32)
            heat_f = heat_u8.astype(np.float32)
            composite = bg_f * (1.0 - alpha_3) + heat_f * alpha_3
            img8 = composite.clip(0, 255).astype(np.uint8)
        else:
            img8 = heat_u8

        return img8

    @staticmethod
    def get_stats(heatmap_np):
        return {
            "heatmap/unique_global_tiles": int(np.count_nonzero(heatmap_np)),
        }

    @staticmethod
    def get_map_table(heatmap_np):
        """Build a list of [map_id, name, tiles_visited, total_visits] for maps with activity."""
        try:
            from pufferlib.ocean.pfr_native import binding
        except ImportError:
            return None, 0

        import re
        lut_path = Path(__file__).resolve().parent / "pfr_heatmap_lut.h"
        if not lut_path.exists():
            return None, 0

        offsets = {}
        for line in lut_path.read_text().splitlines():
            m = re.match(r'\s*\[(\d+)\]\s*=.*gx\s*=\s*(-?\d+).*gy\s*=\s*(-?\d+)', line)
            if m:
                mid, gx, gy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if gx >= 0:
                    offsets[mid] = (gx, gy)

        rows = []
        n_maps = binding.get_map_count()
        for mid in range(n_maps):
            if mid not in offsets:
                continue
            gx, gy = offsets[mid]
            name, w, h, _ = binding.get_map_info(mid)
            y1, y2 = max(0, gy), min(heatmap_np.shape[0], gy + h)
            x1, x2 = max(0, gx), min(heatmap_np.shape[1], gx + w)
            if y1 >= y2 or x1 >= x2:
                continue
            region = heatmap_np[y1:y2, x1:x2]
            nz = int(np.count_nonzero(region))
            total = int(np.sum(region))
            if nz > 0:
                rows.append([mid, name, nz, total])

        rows.sort(key=lambda r: -r[3])
        return rows, len(rows)


def _hsv_to_rgb(h, s, v):
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [v, q, p, p, t, v], default=v)
    g = np.select(conds, [t, v, v, q, p, p], default=v)
    b = np.select(conds, [p, p, t, v, v, q], default=v)

    z = s == 0
    r = np.where(z, v, r)
    g = np.where(z, v, g)
    b = np.where(z, v, b)
    return np.stack([r, g, b], axis=-1)
