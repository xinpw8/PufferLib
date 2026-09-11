from __future__ import annotations

import hashlib
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from gpu_motion_assets import GpuMotionAssets, _normalize_clip_heading_wxyz


def heading_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(
        np.float32(2.0) * (x * y + z * w),
        np.float32(1.0) - np.float32(2.0) * (y * y + z * z),
    ))


class ClipHeadingNormalizationTests(unittest.TestCase):
    def test_frame_zero_yaw_is_removed_without_erasing_relative_yaw(self):
        yaw = np.array([2.342987, 2.3605738], dtype=np.float32)
        roots = np.zeros((2, 4), dtype=np.float32)
        roots[:, 0] = np.cos(yaw * np.float32(0.5))
        roots[:, 3] = np.sin(yaw * np.float32(0.5))

        _normalize_clip_heading_wxyz(roots)

        self.assertLessEqual(abs(heading_wxyz(roots[0])), 2.0e-6)
        self.assertAlmostEqual(
            heading_wxyz(roots[1]), float(yaw[1] - yaw[0]), places=6)

    def test_nonunit_roots_are_rejected(self):
        roots = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unit WXYZ"):
            _normalize_clip_heading_wxyz(roots)


ASSET_ROOT_TEXT = os.environ.get("REK_G1_SEMANTIC_ASSETS")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
@unittest.skipUnless(ASSET_ROOT_TEXT, "REK_G1_SEMANTIC_ASSETS is unset")
class RealSemanticAssetTests(unittest.TestCase):
    def test_gpu_loader_matches_native_heading_normalization_contract(self):
        root = Path(ASSET_ROOT_TEXT)
        manifest = root / "semantic_duel_assets_manifest.json"
        assets = GpuMotionAssets(
            root, hashlib.sha256(manifest.read_bytes()).hexdigest())
        raw_nonzero = 0
        for clip in assets.clips.values():
            wxyz_name = clip["files"]["wxyz"]
            xyzw_name = clip["files"]["xyzw"]
            raw = np.fromfile(root / wxyz_name, dtype="<f4").reshape(-1, 4)
            normalized = assets.host_arrays[wxyz_name]
            raw_nonzero += abs(heading_wxyz(raw[0])) >= 1.0e-6
            self.assertLessEqual(abs(heading_wxyz(normalized[0])), 2.0e-6)
            np.testing.assert_array_equal(
                assets.host_arrays[xyzw_name].view(np.uint32),
                normalized[:, [1, 2, 3, 0]].copy().view(np.uint32),
            )
            torch.testing.assert_close(
                assets.arrays[wxyz_name].cpu(), torch.from_numpy(normalized),
                rtol=0.0, atol=0.0)
        self.assertGreater(raw_nonzero, 0)


if __name__ == "__main__":
    unittest.main()
