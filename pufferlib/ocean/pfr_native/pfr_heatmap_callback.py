"""
pfr_heatmap_callback.py — Exploration heatmap overlay on world map.

Reads the shared heatmap buffer accumulated in C (zero-copy via _C.get_heatmap)
and composites a visit heatmap over the static world_map.png for wandb logging.
No per-step Python overhead — render only on epoch boundaries.
"""

import numpy as np
from pathlib import Path

# World map resource (generated once by tools/gen_world_map.py)
_WORLD_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "pfr_native" / "world_map.png"
_cached_bg = None


def _load_world_map_bg():
    """Load world_map.png as uint8 RGB array (cached)."""
    global _cached_bg
    if _cached_bg is not None:
        return _cached_bg
    try:
        from PIL import Image
        if _WORLD_MAP_PATH.exists():
            img = Image.open(str(_WORLD_MAP_PATH)).convert("RGB")
            _cached_bg = np.array(img, dtype=np.uint8)
        else:
            _cached_bg = None
    except ImportError:
        _cached_bg = None
    return _cached_bg


class PfrHeatmapCallback:
    def __init__(self, render_interval=10):
        self.render_interval = max(1, int(render_interval))
        self._log_calls = 0

    def should_render(self):
        self._log_calls += 1
        return self._log_calls % self.render_interval == 0

    def render(self, heatmap_np):
        """
        Render heatmap overlaid on the world map.

        Args:
            heatmap_np: np.ndarray float32 shape (H, W), accumulated visit counts.

        Returns:
            np.ndarray uint8 shape (H*2, W*2, 3), or None if empty.
        """
        mx = np.max(heatmap_np)
        if mx == 0:
            return None

        bg = _load_world_map_bg()

        log_c = np.log1p(heatmap_np)
        log_mx = np.max(log_c)
        scaled = log_c / log_mx
        nz = (heatmap_np > 0).astype(np.float32)

        # HSV heatmap: blue (low) -> red (high)
        h = 2.0 * (1.0 - scaled) / 3.0
        heat_rgb = _hsv_to_rgb(h, nz, nz)
        heat_u8 = (255 * heat_rgb).astype(np.uint8)

        if bg is not None:
            # Composite: alpha-blend heatmap over world map background
            # Alpha = visit intensity (nonzero pixels get alpha 0.7-1.0)
            alpha = (0.7 * nz + 0.3 * scaled * nz)
            alpha_3 = alpha[:, :, np.newaxis]

            # Place world map into heatmap-sized canvas at (pad, pad) offset.
            # The heatmap LUT adds 'pad' to all coordinates, so the world map
            # (which was rendered without padding) must be offset to match.
            hh, hw = heatmap_np.shape
            bh, bw = bg.shape[:2]
            pad = (hh - bh) // 2 if hh > bh else 0
            comp_bg = np.full((hh, hw, 3), 30, dtype=np.uint8)
            y_end = min(pad + bh, hh)
            x_end = min(pad + bw, hw)
            comp_bg[pad:y_end, pad:x_end] = bg[:y_end - pad, :x_end - pad]

            bg_f = comp_bg.astype(np.float32)
            heat_f = heat_u8.astype(np.float32)
            composite = bg_f * (1.0 - alpha_3) + heat_f * alpha_3
            img8 = composite.clip(0, 255).astype(np.uint8)
        else:
            # No world map available — plain heatmap
            img8 = heat_u8

        # 2x upscale for better visibility
        return np.repeat(np.repeat(img8, 2, axis=0), 2, axis=1)

    @staticmethod
    def get_stats(heatmap_np):
        return {
            "heatmap/unique_global_tiles": int(np.count_nonzero(heatmap_np)),
        }


def _hsv_to_rgb(h, s, v):
    """Vectorized HSV->RGB. All (H,W) float [0,1]. Returns (H,W,3)."""
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
