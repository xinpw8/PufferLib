"""Explicit, opt-in policy coordinates; the simulator's raw ABI is unchanged.

Production training/evaluation activate this only through the explicit policy
encoder option; raw remains the default. Transformed checkpoints must train
and evaluate with the same declared encoder. Reusing pinned raw initial
weights requires its separate, deliberate initialization mode.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch


ENCODER_NAME = "polar_xy_v1"
SCALED_ENCODER_NAME = "scaled_polar_xy_v1"
OBSERVATION_FLOATS = 223
ENCODER_CHOICES = ("raw", ENCODER_NAME, SCALED_ENCODER_NAME)
INITIALIZATION_CHOICES = ("matching-checkpoint", "fresh-random", "raw-initial-weights")
_ENCODER_METADATA = {
    "schema": "rek.g1.policy_observation_encoder.v1",
    "name": ENCODER_NAME,
    "source": "semantic_duel_runtime.h schema 4, 223 binary32 values",
    "input_floats": OBSERVATION_FLOATS,
    "output_floats": OBSERVATION_FLOATS,
    "replaced_policy_columns": {
        "86": "horizontal root-to-opponent distance, metres",
        "87": "opponent bearing relative to projected self-root local +X, radians",
    },
    "unchanged_column_intervals_half_open": [[0, 86], [88, 223]],
    "source_geometry_columns": {"self_xy": [0, 1], "self_quaternion_wxyz": [3, 4, 5, 6], "opponent_xy": [86, 87]},
    "angle_interval": "[-pi, pi]",
    "zero_range_bearing": "0 rad is the canonical polar coordinate; direction at zero range is undefined",
    "calculation": "normalize wxyz and compute planar polar coordinates in float64, store float32",
    "raw_simulator_observations_mutated": False,
}


def encoder_metadata(encoder_name=ENCODER_NAME):
    """Return a fresh, versioned descriptor for explicit checkpoint manifests."""
    if encoder_name not in (ENCODER_NAME, SCALED_ENCODER_NAME):
        raise ValueError("unknown policy observation encoder")
    descriptor = deepcopy(_ENCODER_METADATA)
    if encoder_name == SCALED_ENCODER_NAME:
        descriptor["name"] = encoder_name
        descriptor["replaced_policy_columns"]["87"] = "opponent ego-yaw bearing, half-turns (radians divided by pi)"
        descriptor["angle_interval"] = "[-1, 1] half-turns"
        descriptor["unchanged_column_intervals_half_open"] = [[0, 72], [73, 86], [88, 158], [159, 188], [190, 223]]
        descriptor["fixed_scale_divisors"] = {"72": 180.0, "87": math.pi, "158": 180.0, "188": 120.0, "189": 120.0}
        descriptor["rescaled_policy_columns"] = {
            "72": "self tilt degrees divided by 180, half-turns",
            "158": "opponent tilt degrees divided by 180, half-turns",
            "188": "round duration seconds divided by the fixed 120 s reference",
            "189": "time remaining seconds divided by the fixed 120 s reference",
        }
        descriptor["time_reference_source"] = "g1_fight_state.c REK_G1_FIGHT_CONFIG_F84F1874.normal_round_seconds = 120.0f"
        descriptor["fixed_scaling"] = "float64 division followed by float32 storage; no clipping or adaptive statistics"
    return descriptor


def encoder_fingerprint(encoder_name=ENCODER_NAME):
    payload = json.dumps(encoder_metadata(encoder_name), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def require_checkpoint_encoder_metadata(manifest: Mapping, checkpoint_sha256: str, *, encoder_name=ENCODER_NAME):
    """Check the caller's already-hashed checkpoint against its declared view.

    This function does not relabel a checkpoint or infer its encoder from width.
    A checkpoint reader must compute checkpoint_sha256 from the actual file.
    The manifest must bind that same hash and contain the exact descriptor.
    """
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA256 of the actual checkpoint")
    if not isinstance(manifest, Mapping):
        raise ValueError("an explicitly encoded checkpoint manifest is required")
    checkpoint = manifest.get("checkpoint", {})
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("encoder manifest is not bound to this checkpoint hash")
    if manifest.get("policy_observation_encoder") != encoder_metadata(encoder_name):
        raise ValueError(f"checkpoint does not declare the exact {encoder_name} policy view; other encodings are incompatible")
    if manifest.get("policy_observation_encoder_sha256") != encoder_fingerprint(encoder_name):
        raise ValueError("checkpoint observation-encoder fingerprint mismatch")


def policy_encoder_report(encoder_name, initialization="matching-checkpoint"):
    if encoder_name not in ENCODER_CHOICES or initialization not in INITIALIZATION_CHOICES:
        raise ValueError("unknown policy observation encoder or initialization")
    return {"policy_observation_encoder": ({"name": "raw", "input_floats": 223, "output_floats": 223}
                                            if encoder_name == "raw" else encoder_metadata(encoder_name)),
            "policy_observation_encoder_sha256": None if encoder_name == "raw" else encoder_fingerprint(encoder_name),
            "policy_observation_initialization": initialization}


def load_policy_encoder_checkpoint(path, encoder_name, *, initialization="matching-checkpoint", expected_sha256=None):
    """Validate input provenance before allocating any simulation or learner."""
    policy_encoder_report(encoder_name, initialization)
    if initialization == "fresh-random":
        if path is not None:
            raise ValueError("fresh-random requires no loaded checkpoint")
        return None, None
    if path is None:
        if encoder_name != "raw" or initialization == "raw-initial-weights":
            raise ValueError("transformed initialization requires a checkpoint or explicit fresh-random mode")
        return None, None
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("loaded checkpoint SHA256 mismatch")
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(sidecar.read_text()) if sidecar.exists() else {"checkpoint": {"sha256": actual}}
    if manifest.get("checkpoint", {}).get("sha256") != actual:
        raise ValueError("checkpoint manifest hash does not match the actual weights")
    declared = manifest.get("policy_observation_encoder", {}).get("name", "raw")
    if initialization == "raw-initial-weights":
        if encoder_name == "raw" or expected_sha256 is None or declared != "raw":
            raise ValueError("raw-initial-weights requires an explicitly pinned raw checkpoint and a transformed target")
    elif encoder_name == "raw":
        if declared != "raw":
            raise ValueError("transformed checkpoint cannot be evaluated or resumed through raw observations")
    else:
        require_checkpoint_encoder_metadata(manifest, actual, encoder_name=encoder_name)
    return manifest, actual


def save_policy_weights(trainer, path, encoder_name="raw", initialization="matching-checkpoint"):
    """Annotate every orchestration-owned transformed save, without monkeypatches."""
    manifest = trainer.save_weights(path)
    if encoder_name == "raw":
        return manifest
    path = Path(path)
    if not isinstance(manifest, dict) or manifest.get("checkpoint", {}).get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("native checkpoint save returned inconsistent metadata")
    manifest.update(policy_encoder_report(encoder_name, initialization))
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _root_forward_xy(observations):
    q = observations[:, 3:7].double()
    norm = torch.linalg.vector_norm(q, dim=1)
    w, x, y, z = (q / norm[:, None]).unbind(1)
    forward_x = 1 - 2 * (y*y + z*z)
    forward_y = 2 * (w*z + x*y)
    return forward_x, forward_y, norm


class GpuPolarXYPolicyEncoder:
    """Owned, stable 223-wide view, without changing any environment buffer.

    Only output columns 86 and 87 change meaning. All other columns, including
    self world XY, both quaternions, opponent height, events and scores, are
    copied unchanged. Given the retained self pose, range/bearing recovers the
    original opponent XY up to the binary32 encoding's rounding error.

    Each encode call uses device tensor operations only. Invalid inputs latch
    status and produce no guessed direction; check_status is the explicit
    reporting/synchronization boundary. The CPU switch exists for tests only.
    """

    encoder_name = ENCODER_NAME

    def __init__(self, rows, device, *, checkpoint_manifest=None, checkpoint_sha256=None,
                 initialization="matching-checkpoint", allow_cpu_for_tests=False):
        if initialization == "matching-checkpoint":
            require_checkpoint_encoder_metadata(checkpoint_manifest, checkpoint_sha256, encoder_name=self.encoder_name)
        elif initialization == "fresh-random":
            if checkpoint_manifest is not None or checkpoint_sha256 is not None:
                raise ValueError("fresh-random encoding cannot reuse checkpoint metadata")
        elif initialization == "raw-initial-weights":
            if (not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
                    or not isinstance(checkpoint_manifest, Mapping)
                    or checkpoint_manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha256):
                raise ValueError("raw initialization needs validated source checkpoint provenance")
            if checkpoint_manifest.get("policy_observation_encoder", {}).get("name", "raw") != "raw":
                raise ValueError("raw initialization source must declare raw inputs")
        else:
            raise ValueError("unknown policy initialization")
        if rows < 1:
            raise ValueError("encoder requires a positive row count")
        requested = torch.device(device)
        if requested.type != "cuda" and not (requested.type == "cpu" and allow_cpu_for_tests):
            raise ValueError("policy observation encoding requires CUDA; CPU is test-only")
        self.rows = rows
        self.observations = torch.empty((rows, OBSERVATION_FLOATS), dtype=torch.float32, device=requested)
        self.device = self.observations.device
        self.status = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self.scale_indices = None
        if self.encoder_name == SCALED_ENCODER_NAME:
            divisors = encoder_metadata(self.encoder_name)["fixed_scale_divisors"]
            self.scale_indices = torch.tensor([int(index) for index in divisors], dtype=torch.int64, device=self.device)
            self.scale_divisors = torch.tensor(list(divisors.values()), dtype=torch.float64, device=self.device)

    def reset_status(self):
        self.status.zero_()

    def encode(self, raw_observations):
        if raw_observations.shape != (self.rows, OBSERVATION_FLOATS):
            raise ValueError("raw observations must have shape [rows, 223]")
        if raw_observations.dtype != torch.float32 or raw_observations.device != self.device:
            raise ValueError("raw observations must be float32 on the encoder device")
        if raw_observations.data_ptr() == self.observations.data_ptr():
            raise ValueError("encoded policy view must never be passed as raw observations")
        forward_x, forward_y, norm = _root_forward_xy(raw_observations)
        dx, dy = (raw_observations[:, 86:88].double() - raw_observations[:, :2].double()).unbind(1)
        distance = torch.sqrt(dx*dx + dy*dy)
        forward = forward_x*dx + forward_y*dy
        lateral = -forward_y*dx + forward_x*dy
        bearing = torch.atan2(lateral, forward)
        # At coincident XY, angle contains no source information. Canonical
        # zero retains an invertible representation of the position itself.
        bearing = torch.where(distance == 0, 0.0, bearing)
        invalid_heading = (forward_x*forward_x + forward_y*forward_y) == 0
        invalid_direction = invalid_heading & (distance != 0)
        bearing = torch.where(invalid_direction, float("nan"), bearing)
        self.observations.copy_(raw_observations)
        self.observations[:, 86].copy_(distance)
        self.observations[:, 87].copy_(bearing)
        if self.scale_indices is not None:
            scaled = self.observations.index_select(1, self.scale_indices).double() / self.scale_divisors
            self.observations.index_copy_(1, self.scale_indices, scaled.float())
        self.status.bitwise_or_(
            (~torch.isfinite(raw_observations).all(1)).to(torch.int32)
            | ((~torch.isfinite(norm) | (norm <= 0)).to(torch.int32) * 2)
            | (invalid_direction.to(torch.int32) * 4)
            | ((~torch.isfinite(self.observations).all(1)).to(torch.int32) * 8)
        )
        return self.observations

    def check_status(self):
        values = self.status.detach().cpu().tolist()
        if any(values):
            raise RuntimeError(f"invalid {self.encoder_name} source observation: {values}")


class GpuScaledPolarXYPolicyEncoder(GpuPolarXYPolicyEncoder):
    """Separate opt-in units: tilt/180, polar bearing/pi, duration/remain/120.

    The fixed 120 s denominator comes from the candidate's normal-round config.
    It does not impose a duration: the configured 30 s redo round encodes as 0.25.
    No values are clipped and no new state, reward or physical rule is added.
    """

    encoder_name = SCALED_ENCODER_NAME


def reconstruct_opponent_world_xy(policy_view, *, encoder_name=ENCODER_NAME):
    """Coordinate-inverse diagnostic, not a simulation or observation fallback."""
    if policy_view.ndim != 2 or policy_view.shape[1] != OBSERVATION_FLOATS:
        raise ValueError("policy view must be [rows, 223]")
    fx, fy, _ = _root_forward_xy(policy_view)
    length = torch.sqrt(fx*fx + fy*fy)
    yaw = torch.atan2(fy, fx)
    distance, bearing = policy_view[:, 86].double(), policy_view[:, 87].double()
    if encoder_name == SCALED_ENCODER_NAME:
        bearing = bearing * math.pi
    elif encoder_name != ENCODER_NAME:
        raise ValueError("unknown policy observation encoder")
    dx, dy = distance * torch.cos(yaw + bearing), distance * torch.sin(yaw + bearing)
    result = policy_view[:, :2].double() + torch.stack((dx, dy), dim=1)
    # A vertical root heading cannot identify a nonzero planar direction.
    invalid = (length == 0) & (distance != 0)
    return torch.where(invalid[:, None], float("nan"), result).to(policy_view.dtype)


def reconstruct_raw_observations(policy_view, *, encoder_name=ENCODER_NAME):
    """Test/report inverse retaining source units up to binary32 rounding."""
    descriptor = encoder_metadata(encoder_name)
    result = policy_view.clone()
    for index, divisor in descriptor.get("fixed_scale_divisors", {}).items():
        result[:, int(index)] = (result[:, int(index)].double() * divisor).float()
    result[:, 86:88] = reconstruct_opponent_world_xy(result)
    return result
