"""Load the existing semantic motion bundle once into CUDA storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _normalize_clip_heading_wxyz(values: np.ndarray) -> None:
    """Apply SonicMotionComposer.NormalizeClipHeading in binary32."""
    if values.dtype != np.float32 or values.ndim != 2 \
            or values.shape[0] == 0 or values.shape[1] != 4:
        raise ValueError("clip roots must be a nonempty float32 WXYZ matrix")
    norms = np.sqrt(np.sum(values.astype(np.float64) ** 2, axis=1))
    if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.0) > 1.0e-4):
        raise ValueError("clip roots must contain unit WXYZ quaternions")

    w, x, y, z = values[0]
    cross_sum = np.float32(np.float32(x * y) + np.float32(z * w))
    sum_squares = np.float32(np.float32(y * y) + np.float32(z * z))
    heading = np.arctan2(
        np.float32(np.float32(2.0) * cross_sum),
        np.float32(np.float32(1.0) - np.float32(np.float32(2.0) * sum_squares)),
    )
    if not np.isfinite(heading):
        raise ValueError("clip frame-zero heading must be finite")
    if np.abs(heading) >= np.float32(1.0e-6):
        half_inverse_heading = np.float32(np.float32(-0.5) * heading)
        cosine = np.cos(half_inverse_heading)
        sine = np.sin(half_inverse_heading)
        old = values.copy()
        values[:, 0] = np.float32(old[:, 0] * cosine) \
            - np.float32(old[:, 3] * sine)
        values[:, 1] = np.float32(old[:, 1] * cosine) \
            - np.float32(old[:, 2] * sine)
        values[:, 2] = np.float32(old[:, 1] * sine) \
            + np.float32(old[:, 2] * cosine)
        values[:, 3] = np.float32(old[:, 3] * cosine) \
            + np.float32(old[:, 0] * sine)

    norms = np.sqrt(np.sum(values.astype(np.float64) ** 2, axis=1))
    if not np.isfinite(values).all() or np.any(np.abs(norms - 1.0) > 1.0e-4):
        raise ValueError("normalized clip roots are invalid")


class GpuMotionAssets:
    def __init__(self, root: Path, manifest_sha256: str, device: str = "cuda:0"):
        if torch.device(device).type != "cuda":
            raise ValueError("motion storage requires CUDA")
        self.root = root
        payload = (root / "semantic_duel_assets_manifest.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest_sha256:
            raise ValueError("semantic asset manifest SHA-256 mismatch")
        self.manifest = json.loads(payload)
        if self.manifest["schema"] != "rek.g1_semantic_duel_assets.v1":
            raise ValueError("unsupported semantic asset schema")
        self.manifest_sha256 = manifest_sha256
        self.routes = self.manifest["routes"]
        if [route["route_id"] for route in self.routes] != list(range(24)):
            raise ValueError("expected the existing 24 motion routes")
        self.arrays = {}
        self.host_arrays = {}
        for name, record in self.manifest["files"].items():
            if Path(name).name != name:
                raise ValueError("asset filename must be relative to its bundle")
            raw = (root / name).read_bytes()
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"semantic asset identity mismatch: {name}")
            if record.get("dtype") == "float32_le":
                values = np.frombuffer(raw, dtype="<f4").reshape(record["shape"]).copy()
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite semantic asset: {name}")
                self.host_arrays[name] = values
        self.clips = {
            clip["npz_path_id"]: clip for clip in self.manifest["clips"]
        }
        for clip in self.clips.values():
            wxyz_name = clip["files"]["wxyz"]
            xyzw_name = clip["files"]["xyzw"]
            wxyz = self.host_arrays[wxyz_name]
            xyzw = self.host_arrays[xyzw_name]
            if xyzw.shape != wxyz.shape:
                raise ValueError(f"clip root storage shape mismatch: {wxyz_name}")
            _normalize_clip_heading_wxyz(wxyz)
            xyzw[:] = wxyz[:, [1, 2, 3, 0]]
        self.arrays = {
            name: torch.as_tensor(values, device=device)
            for name, values in self.host_arrays.items()
        }
        self.roles = {clip["role"]: clip for clip in self.clips.values()}
        self.future_offsets = torch.arange(10, device=device) * 5

    def fixed_reference(self, role: str, cursors: torch.Tensor, *, loop: bool):
        """Native fixed-clip reference windows for component validation.

        Semantic actions must use the complete compositor instead of this
        fixed-clip helper. A cursor denotes the current frame for each robot.
        """
        clip = self.roles[role]
        position = self.arrays[clip["files"]["mujoco_joint_order"]]
        rotation = self.arrays[clip["files"]["xyzw"]]
        if cursors.device != position.device:
            raise ValueError("motion cursors must share the CUDA asset device")
        indices = cursors[:, None] + self.future_offsets[None, :]
        if loop:
            indices = indices.remainder(clip["frames"])
            next_indices = (indices + 1).remainder(clip["frames"])
        else:
            indices = indices.clamp(0, clip["frames"] - 1)
            next_indices = (indices + 1).clamp(0, clip["frames"] - 1)
        return position[indices], position[next_indices], rotation[indices]
