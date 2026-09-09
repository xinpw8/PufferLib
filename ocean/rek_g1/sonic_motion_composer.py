"""Measured REK G1 Sonic motion-composer semantics.

This module implements only the behavior recovered directly from the current
``REKApp.SonicMotionComposer`` native dump:

* ``InstallLayer`` at native VA 0x182295660
* ``Advance`` at native VA 0x182293840
* ``BuildMirrorTables`` at native VA 0x1822945D0
* ``CancelAction`` at native VA 0x182294910
* ``ConsumeHeadingDelta`` at native VA 0x182294B50
* ``CopyLayer`` at native VA 0x182294B70
* ``GetReferenceFrame`` at native VA 0x182295060
* ``LayerHeadingContribution`` at native VA 0x182295A80
* ``LayerRootHeading`` at native VA 0x182295C40
* ``PlayActionImmediate`` at native VA 0x182298200
* ``PlayAction`` at native VA 0x182298240
* ``SampleLayer`` at native VA 0x182299F40
* ``SetLocomotionSpeed`` at native VA 0x18229A830
* ``WeightCurrent`` at native VA 0x18229AED0
* ``WrapLoopCursor`` at native VA 0x18229AFE0
* ``WrapPi`` at native VA 0x18229B070
* ``XfadeAt`` at native VA 0x18229B0B0
* ``ResolveFrames`` at native VA 0x182299020
* deduplicated ``CalcHeadingWxyz`` at native VA 0x18227A310

The current build dispatches quaternion interpolation through Unity's runtime
``Quaternion.Internal_Slerp`` icall. Active-source loop entry selection also
uses a feature matcher. Those two implementations are not statically present
at the recovered call sites, so APIs that need them require injected exact
backends and fail closed when none is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import struct
from typing import Any, Callable, Mapping, NamedTuple, Sequence


CONTRACT_SCHEMA = "rek.g1_motion_clip_contract.v1"


class UnsupportedComposerSemantics(NotImplementedError):
    """Raised when a caller requests composer behavior not implemented here."""


def _f32(value: float) -> float:
    """Round an operation result to the binary32 precision used by the game."""

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


MIN_ABS_PLAYBACK_SPEED = _f32(0.01)
MIN_LOCOMOTION_SCALE = _f32(0.05)
HALF = _f32(0.5)
PI_F32 = _f32(3.1415927)
NEG_PI_F32 = _f32(-3.1415927)
TWO_PI_F32 = _f32(6.2831855)
NEG_TWO_PI_F32 = _f32(-6.2831855)


@dataclass(frozen=True)
class NpzClip:
    frame_count: int
    fps: float
    name: str = ""
    filename: str = ""
    sha256: str = ""
    path_id: int | None = None

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be finite and positive")


@dataclass(frozen=True)
class MocapClipConfig:
    mirror: bool
    loop: bool
    playback_speed: float
    start_frame: int
    end_frame: int
    path_id: int | None = None
    blend_in_time: float = _f32(0.1)
    blend_out_time: float = _f32(0.1)
    yaw_blend: float = 0.0


@dataclass(frozen=True)
class ContractClip:
    role: str
    clip: NpzClip
    config: MocapClipConfig


@dataclass
class Layer:
    clip: NpzClip | None = None
    config: MocapClipConfig | None = None
    mirror: bool = False
    loop: bool = False
    speed: float = 1.0
    per_tick: float = 0.0
    cursor: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    feature_instance_id: int | None = None
    on_complete: Callable[[], None] | None = field(default=None, repr=False)
    active: bool = False
    prev_heading: float = 0.0
    last_heading_delta: float = 0.0
    heading_valid: bool = False
    heading_resync: bool = False


class ResolvedFrames(NamedTuple):
    f0: int
    f1: int
    t: float


class AdvanceResult(NamedTuple):
    wrapped: bool
    completed: bool


class LayerRootHeadingResult(NamedTuple):
    heading: float
    seam: bool


class ComposerAdvanceResult(NamedTuple):
    current: AdvanceResult
    outgoing: AdvanceResult
    weight_current: float


Wxyz = tuple[float, float, float, float]
QuaternionSlerp = Callable[[Wxyz, Wxyz, float], Sequence[float]]
Atan2F = Callable[[float, float], float]
SinCosF = Callable[[float], tuple[float, float]]
EntryMatcher = Callable[[Layer, Layer], float]
RootHeadingSampler = Callable[[Layer], LayerRootHeadingResult]


@dataclass(frozen=True)
class ClipSamples:
    """Decoded native NPZ samples, with roots stored in WXYZ order."""

    joint_positions: Sequence[Sequence[float]]
    root_quaternions_wxyz: Sequence[Sequence[float]]


@dataclass(frozen=True)
class PoseSample:
    joint_positions: tuple[float, ...]
    root_quaternion_wxyz: Wxyz


@dataclass(frozen=True)
class MirrorTables:
    source_indices: tuple[int, ...]
    negate: tuple[bool, ...]


def load_contract_clip(
    contract_path: str | Path, role: str
) -> ContractClip:
    """Load one measured clip descriptor from the versioned evidence contract."""

    payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"unsupported motion contract schema: {payload.get('schema')!r}")
    for entry in payload.get("clips", []):
        if entry.get("role") != role:
            continue
        npz = entry["npz"]
        config = entry["mocap_clip_config"]
        return ContractClip(
            role=role,
            clip=NpzClip(
                frame_count=int(npz["frames"]),
                fps=float(npz["fps"]),
                name=str(npz.get("name", "")),
                filename=str(npz.get("output", "")),
                sha256=str(npz.get("sha256", "")),
                path_id=int(npz["path_id"]),
            ),
            config=MocapClipConfig(
                mirror=bool(config["mirror"]),
                loop=bool(config["loop"]),
                playback_speed=float(config["playbackSpeed"]),
                start_frame=int(config["startFrame"]),
                end_frame=int(config["endFrame"]),
                path_id=int(config["path_id"]),
                blend_in_time=float(config["blendInTime"]),
                blend_out_time=float(config["blendOutTime"]),
                yaw_blend=float(config["yawBlend"]),
            ),
        )
    raise KeyError(f"motion contract has no role {role!r}")


def sanitize_playback_speed(speed: float) -> float:
    """Match ``SanitizePlaybackSpeed`` for finite inputs."""

    speed = _f32(speed)
    if not math.isfinite(speed):
        raise UnsupportedComposerSemantics(
            "native non-finite playback-speed behavior is not represented"
        )
    magnitude = max(_f32(MIN_ABS_PLAYBACK_SPEED), abs(speed))
    return _f32(-magnitude if speed < 0.0 else magnitude)


def entry_cursor(layer: Layer) -> float:
    """Select start for nonnegative playback and end for reverse playback."""

    return float(layer.start_frame if layer.per_tick >= 0.0 else layer.end_frame)


def install_layer(
    layer: Layer,
    clip: NpzClip,
    config: MocapClipConfig | None,
    *,
    mirror: bool,
    loop: bool,
    speed: float,
    start_frame: int,
    end_frame: int,
    controller_rate_hz: int = 50,
) -> None:
    """Install a layer using the native clamp, rate, and entry-cursor rules."""

    if controller_rate_hz <= 0:
        raise ValueError("controller_rate_hz must be positive")

    native_speed = sanitize_playback_speed(speed)
    last_frame = clip.frame_count - 1
    clamped_start = 0 if start_frame < 0 else min(start_frame, last_frame)
    requested_end = last_frame if end_frame < 0 else end_frame
    clamped_end = (
        clamped_start
        if requested_end < clamped_start
        else min(requested_end, last_frame)
    )

    layer.mirror = bool(mirror)
    layer.loop = bool(loop)
    layer.clip = clip
    layer.config = config
    layer.speed = native_speed
    fps_over_rate = _f32(_f32(clip.fps) / _f32(controller_rate_hz))
    layer.per_tick = _f32(fps_over_rate * native_speed)
    layer.start_frame = clamped_start
    layer.end_frame = clamped_end
    layer.feature_instance_id = clip.path_id if config is not None else None
    layer.active = True
    layer.heading_valid = False
    layer.heading_resync = False
    layer.cursor = entry_cursor(layer)


def install_contract_clip(
    layer: Layer, contract_clip: ContractClip, *, controller_rate_hz: int = 50
) -> None:
    """Install the exact layer arguments carried by a contract entry."""

    config = contract_clip.config
    install_layer(
        layer,
        contract_clip.clip,
        config,
        mirror=config.mirror,
        loop=config.loop,
        speed=config.playback_speed,
        start_frame=config.start_frame,
        end_frame=config.end_frame,
        controller_rate_hz=controller_rate_hz,
    )


def wrap_loop_cursor(layer: Layer) -> bool:
    """Mutate a loop cursor into the inclusive native frame interval."""

    span = layer.end_frame - layer.start_frame + 1
    if span <= 0:
        return False
    wrapped = False
    upper_exclusive = float(layer.end_frame + 1)
    span_f = _f32(span)
    while layer.cursor >= upper_exclusive:
        layer.cursor = _f32(layer.cursor - span_f)
        wrapped = True
    while layer.cursor < float(layer.start_frame):
        layer.cursor = _f32(layer.cursor + span_f)
        wrapped = True
    return wrapped


def _wrapped_cursor(layer: Layer, cursor: float) -> float:
    span = layer.end_frame - layer.start_frame + 1
    if span <= 0:
        return float(layer.start_frame)
    delta = _f32(cursor - _f32(layer.start_frame))
    remainder = _f32(math.fmod(delta, _f32(span)))
    if remainder < 0.0:
        remainder = _f32(remainder + _f32(span))
    return _f32(_f32(layer.start_frame) + remainder)


def resolve_frames(layer: Layer, frames_ahead: int = 0) -> ResolvedFrames:
    """Resolve the two sampled frames and their linear interpolation fraction."""

    if layer.clip is None:
        raise ValueError("layer has no clip")

    ahead = _f32(_f32(frames_ahead) * layer.per_tick)
    cursor = _f32(layer.cursor + ahead)
    if layer.loop:
        cursor = _wrapped_cursor(layer, cursor)
        f0 = math.floor(cursor)
        f1 = f0 + 1 if f0 + 1 <= layer.end_frame else layer.start_frame
    else:
        cursor = max(float(layer.start_frame), min(cursor, float(layer.end_frame)))
        cursor = _f32(cursor)
        f0 = math.floor(cursor)
        f1 = min(f0 + 1, layer.end_frame)

    t = _f32(cursor - _f32(f0))
    final_clip_index = layer.clip.frame_count - 1
    f0 = max(0, min(f0, final_clip_index))
    f1 = max(0, min(f1, final_clip_index))
    return ResolvedFrames(f0, f1, t)


def advance_layer(layer: Layer) -> AdvanceResult:
    """Advance one layer tick and report loop wrapping or endpoint completion."""

    if not layer.active:
        return AdvanceResult(False, False)

    layer.cursor = _f32(layer.cursor + layer.per_tick)
    if layer.loop:
        return AdvanceResult(wrap_loop_cursor(layer), False)

    layer.cursor = _f32(
        max(float(layer.start_frame), min(layer.cursor, float(layer.end_frame)))
    )
    if layer.per_tick >= 0.0:
        completed = not (layer.cursor < float(layer.end_frame))
    else:
        completed = not (float(layer.start_frame) < layer.cursor)
    return AdvanceResult(False, completed)


def _clamp01(value: float) -> float:
    value = _f32(value)
    if not math.isfinite(value):
        raise UnsupportedComposerSemantics("native non-finite blend behavior is not represented")
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def _blend_frames(controller_rate_hz: int, seconds: float) -> int:
    seconds = _f32(seconds)
    if not math.isfinite(seconds):
        raise UnsupportedComposerSemantics(
            "native non-finite blend-duration behavior is not represented"
        )
    product = _f32(_f32(controller_rate_hz) * seconds)
    rounded = round(product)
    if rounded < -0x80000000 or rounded > 0x7FFFFFFF:
        raise UnsupportedComposerSemantics(
            "native out-of-range RoundToInt behavior is not represented"
        )
    return max(1, int(rounded))


def _lerp_f32(a: float, b: float, t: float) -> float:
    a = _f32(a)
    b = _f32(b)
    t = _clamp01(t)
    return _f32(a + _f32(_f32(b - a) * t))


def _coerce_wxyz(value: Sequence[float], *, source: str) -> Wxyz:
    if len(value) != 4:
        raise ValueError(f"{source} must contain exactly four WXYZ values")
    result = tuple(_f32(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise UnsupportedComposerSemantics(f"non-finite quaternion from {source}")
    return result  # type: ignore[return-value]


def copy_layer(source: Layer, destination: Layer) -> None:
    """Match native ``CopyLayer``, including its forced-active destination."""

    destination.clip = source.clip
    destination.config = source.config
    destination.mirror = source.mirror
    destination.loop = source.loop
    destination.speed = source.speed
    destination.per_tick = source.per_tick
    destination.cursor = source.cursor
    destination.start_frame = source.start_frame
    destination.end_frame = source.end_frame
    destination.feature_instance_id = source.feature_instance_id
    destination.on_complete = None
    destination.active = True
    destination.prev_heading = source.prev_heading
    destination.last_heading_delta = source.last_heading_delta
    destination.heading_valid = source.heading_valid
    destination.heading_resync = source.heading_resync


def build_mirror_tables(joint_names: Sequence[str | None]) -> MirrorTables:
    """Build the exact ASCII mirror permutation and roll/yaw sign table."""

    lookup: dict[str, int] = {}
    normalized: list[str] = []
    for index, raw_name in enumerate(joint_names):
        name = "" if raw_name is None else raw_name
        if not isinstance(name, str):
            raise TypeError("joint names must be strings or None")
        if not name.isascii():
            raise UnsupportedComposerSemantics(
                "non-ASCII ToLowerInvariant mirror semantics are not represented"
            )
        normalized.append(name)
        if name:
            lookup[name] = index

    source_indices: list[int] = []
    negate: list[bool] = []
    for index, name in enumerate(normalized):
        if name.startswith("left_"):
            partner = "right_" + name[5:]
        elif name.startswith("right_"):
            partner = "left_" + name[6:]
        else:
            partner = name
        source_indices.append(lookup.get(partner, index))
        lowered = name.lower()
        negate.append("roll" in lowered or "yaw" in lowered)
    return MirrorTables(tuple(source_indices), tuple(negate))


def weight_current(tt: float, w_in: int, w_out: int) -> float:
    """Evaluate native ``WeightCurrent`` with binary32 operation boundaries."""

    if w_in <= 0 or w_out <= 0:
        raise ValueError("blend widths must be positive")
    tt = _f32(tt)
    if not math.isfinite(tt):
        raise UnsupportedComposerSemantics("native non-finite blend time is not represented")
    outgoing_progress = _clamp01(_f32(tt / _f32(w_out)))
    incoming_progress = _clamp01(_f32(tt / _f32(w_in)))
    denominator = _f32(_f32(1.0 - outgoing_progress) + incoming_progress)
    if denominator <= 0.0:
        return 1.0
    return _f32(incoming_progress / denominator)


def xfade_at(xt: int, frames_ahead: int, w_in: int, w_out: int) -> float:
    """Evaluate ``XfadeAt`` after the native signed-int32 addition."""

    tt = _f32(_i32(int(xt) + int(frames_ahead)))
    return weight_current(tt, w_in, w_out)


def calc_heading_wxyz(quaternion: Sequence[float], *, atan2f: Atan2F | None) -> float:
    """Calculate root yaw using the recovered formula and an exact atan2 backend."""

    if atan2f is None:
        raise UnsupportedComposerSemantics(
            "CalcHeadingWxyz requires the current-build atan2 backend"
        )
    w, x, y, z = _coerce_wxyz(quaternion, source="root quaternion")
    numerator = _f32(
        _f32(_f32(y * x) + _f32(z * w)) + _f32(_f32(y * x) + _f32(z * w))
    )
    sum_squares = _f32(_f32(z * z) + _f32(y * y))
    denominator = _f32(1.0 - _f32(sum_squares + sum_squares))
    heading = _f32(atan2f(numerator, denominator))
    if not math.isfinite(heading):
        raise UnsupportedComposerSemantics("non-finite result from atan2 backend")
    return heading


def wrap_pi(angle: float) -> float:
    """Wrap a finite binary32 angle into the native inclusive pi interval."""

    angle = _f32(angle)
    if not math.isfinite(angle):
        raise UnsupportedComposerSemantics("native non-finite WrapPi behavior is not represented")
    while angle > PI_F32:
        angle = _f32(angle + NEG_TWO_PI_F32)
    while angle < NEG_PI_F32:
        angle = _f32(angle + TWO_PI_F32)
    return angle


def _slerp_wxyz(
    a: Sequence[float],
    b: Sequence[float],
    t: float,
    quaternion_slerp: QuaternionSlerp | None,
) -> Wxyz:
    if quaternion_slerp is None:
        raise UnsupportedComposerSemantics(
            "pose sampling requires the current-build Quaternion.Internal_Slerp backend"
        )
    qa = _coerce_wxyz(a, source="root sample")
    qb = _coerce_wxyz(b, source="root sample")
    return _coerce_wxyz(
        quaternion_slerp(qa, qb, _f32(t)), source="quaternion slerp backend"
    )


def _sample_root_wxyz(
    layer: Layer,
    samples: ClipSamples,
    frames: ResolvedFrames,
    quaternion_slerp: QuaternionSlerp | None,
) -> Wxyz:
    if layer.clip is None:
        raise ValueError("layer has no clip")
    if len(samples.root_quaternions_wxyz) < layer.clip.frame_count:
        raise ValueError("root quaternion samples are shorter than clip.frame_count")
    root = _slerp_wxyz(
        samples.root_quaternions_wxyz[frames.f0],
        samples.root_quaternions_wxyz[frames.f1],
        frames.t,
        quaternion_slerp,
    )
    if layer.mirror:
        root = (root[0], _f32(-root[1]), root[2], _f32(-root[3]))
    return root


def _remove_sampled_yaw(
    root: Wxyz,
    yaw_blend: float,
    *,
    atan2f: Atan2F | None,
    sin_cos_f: SinCosF | None,
) -> Wxyz:
    yaw_blend = _f32(yaw_blend)
    if not math.isfinite(yaw_blend):
        raise UnsupportedComposerSemantics("native non-finite yawBlend behavior is not represented")
    if yaw_blend <= 0.0:
        return root
    if sin_cos_f is None:
        raise UnsupportedComposerSemantics(
            "yaw removal requires the current-build sin/cos backend"
        )
    heading = calc_heading_wxyz(root, atan2f=atan2f)
    half_angle = _f32(_f32(_f32(-heading) * yaw_blend) * HALF)
    sine, cosine = (_f32(value) for value in sin_cos_f(half_angle))
    if not math.isfinite(sine) or not math.isfinite(cosine):
        raise UnsupportedComposerSemantics("non-finite result from sin/cos backend")
    w, x, y, z = root
    return (
        _f32(_f32(w * cosine) - _f32(z * sine)),
        _f32(_f32(x * cosine) - _f32(y * sine)),
        _f32(_f32(x * sine) + _f32(y * cosine)),
        _f32(_f32(z * cosine) + _f32(w * sine)),
    )


def sample_layer(
    layer: Layer,
    samples: ClipSamples,
    frames_ahead: int = 0,
    *,
    mirror_tables: MirrorTables | None = None,
    quaternion_slerp: QuaternionSlerp | None = None,
    atan2f: Atan2F | None = None,
    sin_cos_f: SinCosF | None = None,
    num_dofs: int | None = None,
) -> PoseSample:
    """Sample one layer, including DOF/root mirroring and configured yaw removal."""

    if layer.clip is None:
        raise ValueError("layer has no clip")
    frames = resolve_frames(layer, frames_ahead)
    if len(samples.joint_positions) < layer.clip.frame_count:
        raise ValueError("joint samples are shorter than clip.frame_count")
    row0 = samples.joint_positions[frames.f0]
    row1 = samples.joint_positions[frames.f1]
    if num_dofs is None:
        num_dofs = len(row0)
    if num_dofs < 0 or len(row0) < num_dofs or len(row1) < num_dofs:
        raise ValueError("joint sample row is shorter than num_dofs")
    if layer.mirror:
        if mirror_tables is None:
            raise UnsupportedComposerSemantics("mirrored sampling requires mirror tables")
        if len(mirror_tables.source_indices) != num_dofs or len(mirror_tables.negate) != num_dofs:
            raise ValueError("mirror table length does not match num_dofs")

    positions: list[float] = []
    for output_index in range(num_dofs):
        source_index = (
            mirror_tables.source_indices[output_index]
            if layer.mirror and mirror_tables is not None
            else output_index
        )
        if source_index < 0 or source_index >= len(row0) or source_index >= len(row1):
            raise ValueError("mirror source index is outside the joint sample row")
        value = _lerp_f32(row0[source_index], row1[source_index], frames.t)
        if layer.mirror and mirror_tables is not None and mirror_tables.negate[output_index]:
            value = _f32(-value)
        positions.append(value)

    root = _sample_root_wxyz(layer, samples, frames, quaternion_slerp)
    yaw_blend = layer.config.yaw_blend if layer.config is not None else 0.0
    root = _remove_sampled_yaw(
        root, yaw_blend, atan2f=atan2f, sin_cos_f=sin_cos_f
    )
    return PoseSample(tuple(positions), root)


def layer_root_heading(
    layer: Layer,
    samples: ClipSamples,
    *,
    quaternion_slerp: QuaternionSlerp | None,
    atan2f: Atan2F | None,
) -> LayerRootHeadingResult:
    """Sample the unmodified layer root heading used by heading accumulation."""

    frames = resolve_frames(layer, 0)
    root = _sample_root_wxyz(layer, samples, frames, quaternion_slerp)
    return LayerRootHeadingResult(
        calc_heading_wxyz(root, atan2f=atan2f),
        bool(layer.loop and frames.f1 < frames.f0),
    )


def layer_heading_contribution(
    layer: Layer,
    wrapped: bool,
    *,
    root_heading_sampler: RootHeadingSampler | None,
) -> float:
    """Update a layer's seam-resilient heading state and return its contribution."""

    if layer.config is None:
        layer.heading_valid = False
        return 0.0
    yaw_blend = _f32(layer.config.yaw_blend)
    if not math.isfinite(yaw_blend):
        raise UnsupportedComposerSemantics("native non-finite yawBlend behavior is not represented")
    if yaw_blend <= 0.0:
        layer.heading_valid = False
        return 0.0
    if root_heading_sampler is None:
        raise UnsupportedComposerSemantics(
            "heading accumulation requires an exact layer-root-heading sampler"
        )

    sampled = root_heading_sampler(layer)
    heading = _f32(sampled.heading)
    if not math.isfinite(heading):
        raise UnsupportedComposerSemantics("non-finite root heading sample")
    delta = 0.0
    if layer.heading_valid:
        if sampled.seam or wrapped:
            layer.heading_resync = True
            delta = layer.last_heading_delta
        elif layer.heading_resync:
            layer.heading_resync = False
            delta = layer.last_heading_delta
        else:
            delta = wrap_pi(_f32(heading - layer.prev_heading))
            layer.last_heading_delta = delta
    layer.heading_valid = True
    layer.prev_heading = heading
    return _f32(_f32(delta) * yaw_blend)


def get_reference_frame(
    current_layer: Layer,
    from_layer: Layer,
    frames_ahead: int,
    *,
    samples_for_clip: Callable[[NpzClip], ClipSamples],
    num_dofs: int,
    w_in: int,
    w_out: int,
    xt: int,
    mirror_tables: MirrorTables | None = None,
    quaternion_slerp: QuaternionSlerp | None = None,
    atan2f: Atan2F | None = None,
    sin_cos_f: SinCosF | None = None,
) -> PoseSample:
    """Match ``GetReferenceFrame`` for inactive, single, and blended layers."""

    if num_dofs < 0:
        raise ValueError("num_dofs must be nonnegative")
    if not current_layer.active:
        return PoseSample(tuple(0.0 for _ in range(num_dofs)), (1.0, 0.0, 0.0, 0.0))
    if current_layer.clip is None:
        raise ValueError("active current layer has no clip")
    current = sample_layer(
        current_layer,
        samples_for_clip(current_layer.clip),
        frames_ahead,
        mirror_tables=mirror_tables,
        quaternion_slerp=quaternion_slerp,
        atan2f=atan2f,
        sin_cos_f=sin_cos_f,
        num_dofs=num_dofs,
    )
    if not from_layer.active:
        return current
    if from_layer.clip is None:
        raise ValueError("active outgoing layer has no clip")
    outgoing = sample_layer(
        from_layer,
        samples_for_clip(from_layer.clip),
        frames_ahead,
        mirror_tables=mirror_tables,
        quaternion_slerp=quaternion_slerp,
        atan2f=atan2f,
        sin_cos_f=sin_cos_f,
        num_dofs=num_dofs,
    )
    weight = xfade_at(xt, frames_ahead, w_in, w_out)
    positions = tuple(
        _lerp_f32(old, new, weight)
        for old, new in zip(outgoing.joint_positions, current.joint_positions)
    )
    root = _slerp_wxyz(
        outgoing.root_quaternion_wxyz,
        current.root_quaternion_wxyz,
        weight,
        quaternion_slerp,
    )
    return PoseSample(positions, root)


@dataclass
class TransitionComposer:
    """Recovered two-layer transition, completion, and heading state machine."""

    controller_rate_hz: int = 50
    current_layer: Layer = field(default_factory=Layer)
    from_layer: Layer = field(default_factory=Layer)
    xt: int = 0
    w_in: int = 1
    w_out: int = 1
    w_total: int = 1
    action_playing: bool = False
    action_move_id: int = 0
    pending_heading_delta: float = 0.0
    entry_matcher: EntryMatcher | None = field(default=None, repr=False)
    root_heading_sampler: RootHeadingSampler | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.controller_rate_hz <= 0:
            raise ValueError("controller_rate_hz must be positive")
        if self.w_in <= 0 or self.w_out <= 0 or self.w_total <= 0:
            raise ValueError("blend widths must be positive")
        self.xt = _i32(self.xt)
        self.pending_heading_delta = _f32(self.pending_heading_delta)

    def weight_current(self, tt: float) -> float:
        return weight_current(tt, self.w_in, self.w_out)

    def xfade_at(self, frames_ahead: int) -> float:
        return xfade_at(self.xt, frames_ahead, self.w_in, self.w_out)

    def play_action(
        self,
        clip: NpzClip | None,
        config: MocapClipConfig | None,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """Install an action and preserve the old current layer for blending."""

        if clip is None or config is None:
            return False

        outgoing_active = self.current_layer.active
        if config.loop and outgoing_active and self.entry_matcher is None:
            raise UnsupportedComposerSemantics(
                "active-source loop entry requires the current-build feature matcher"
            )

        new_w_in = self.w_in
        new_w_out = self.w_out
        new_w_total = self.w_total
        if outgoing_active:
            new_w_in = _blend_frames(self.controller_rate_hz, config.blend_in_time)
            outgoing_blend = (
                self.current_layer.config.blend_out_time
                if self.current_layer.config is not None
                else 0.0
            )
            new_w_out = _blend_frames(self.controller_rate_hz, outgoing_blend)
            new_w_total = max(new_w_in, new_w_out)

        if outgoing_active:
            copy_layer(self.current_layer, self.from_layer)
        else:
            self.from_layer.active = False
        self.xt = 0
        install_layer(
            self.current_layer,
            clip,
            config,
            mirror=config.mirror,
            loop=config.loop,
            speed=config.playback_speed,
            start_frame=config.start_frame,
            end_frame=config.end_frame,
            controller_rate_hz=self.controller_rate_hz,
        )
        self.current_layer.on_complete = on_complete
        self.action_playing = not config.loop
        if not config.loop:
            self.action_move_id = _i32(self.action_move_id + 1)

        self.w_in = new_w_in
        self.w_out = new_w_out
        self.w_total = new_w_total
        if config.loop and outgoing_active:
            assert self.entry_matcher is not None
            matched = _f32(self.entry_matcher(self.current_layer, self.from_layer))
            if not math.isfinite(matched):
                raise UnsupportedComposerSemantics("non-finite cursor from entry matcher")
            if matched < self.current_layer.start_frame or matched > self.current_layer.end_frame:
                raise UnsupportedComposerSemantics("out-of-range cursor from entry matcher")
            self.current_layer.cursor = matched
            self.current_layer.heading_valid = False
        else:
            self.current_layer.cursor = entry_cursor(self.current_layer)
        self.current_layer.heading_resync = False
        return True

    def play_action_immediate(
        self,
        clip: NpzClip | None,
        config: MocapClipConfig | None,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        if not self.play_action(clip, config, on_complete):
            return False
        self.from_layer.active = False
        self.xt = 0
        return True

    def cancel_action(self) -> None:
        self.from_layer.active = False
        self.current_layer.on_complete = None
        self.action_playing = False
        self.xt = 0
        self.current_layer.heading_valid = False
        self.current_layer.heading_resync = False

    def set_locomotion_speed(self, scale: float) -> None:
        layer = self.current_layer
        if not layer.active or not layer.loop:
            return
        if layer.clip is None:
            raise ValueError("active loop layer has no clip")
        scale = _f32(scale)
        if not math.isfinite(scale):
            raise UnsupportedComposerSemantics(
                "native non-finite locomotion scale behavior is not represented"
            )
        fps_over_rate = _f32(_f32(layer.clip.fps) / _f32(self.controller_rate_hz))
        base = _f32(fps_over_rate * layer.speed)
        layer.per_tick = _f32(base * max(MIN_LOCOMOTION_SCALE, scale))

    def _require_heading_backend_for_active_layers(self) -> None:
        layers_that_will_contribute = [self.current_layer]
        if self.from_layer.active and _i32(self.xt + 1) < self.w_total:
            layers_that_will_contribute.append(self.from_layer)
        for layer in layers_that_will_contribute:
            if not layer.active or layer.config is None:
                continue
            yaw_blend = _f32(layer.config.yaw_blend)
            if not math.isfinite(yaw_blend):
                raise UnsupportedComposerSemantics(
                    "native non-finite yawBlend behavior is not represented"
                )
            if yaw_blend > 0.0 and self.root_heading_sampler is None:
                raise UnsupportedComposerSemantics(
                    "heading accumulation requires an exact layer-root-heading sampler"
                )

    def advance(self) -> ComposerAdvanceResult:
        self._require_heading_backend_for_active_layers()

        current_result = advance_layer(self.current_layer)
        if current_result.completed:
            self.action_playing = False
            callback = self.current_layer.on_complete
            self.current_layer.on_complete = None
            if callback is not None:
                callback()

        outgoing_result = AdvanceResult(False, False)
        if self.from_layer.active:
            outgoing_result = advance_layer(self.from_layer)
            self.xt = _i32(self.xt + 1)
            if self.xt >= self.w_total:
                self.from_layer.active = False

        current_weight = self.weight_current(float(self.xt)) if self.from_layer.active else 1.0
        if self.current_layer.active:
            contribution = layer_heading_contribution(
                self.current_layer,
                current_result.wrapped,
                root_heading_sampler=self.root_heading_sampler,
            )
            self.pending_heading_delta = _f32(
                self.pending_heading_delta + _f32(contribution * current_weight)
            )
        if self.from_layer.active:
            contribution = layer_heading_contribution(
                self.from_layer,
                outgoing_result.wrapped,
                root_heading_sampler=self.root_heading_sampler,
            )
            outgoing_weight = _f32(1.0 - current_weight)
            self.pending_heading_delta = _f32(
                self.pending_heading_delta + _f32(contribution * outgoing_weight)
            )
        return ComposerAdvanceResult(current_result, outgoing_result, current_weight)

    def consume_heading_delta(self) -> float:
        result = self.pending_heading_delta
        self.pending_heading_delta = 0.0
        return result

    @staticmethod
    def _yaw_ownership(layer: Layer) -> float:
        if not layer.active or layer.config is None:
            return 0.0
        yaw_blend = _f32(layer.config.yaw_blend)
        if not math.isfinite(yaw_blend):
            raise UnsupportedComposerSemantics(
                "native non-finite yawBlend behavior is not represented"
            )
        return _clamp01(max(0.0, yaw_blend))

    @property
    def heading_clip_ownership(self) -> float:
        weight = self.weight_current(float(self.xt)) if self.from_layer.active else 1.0
        current = _f32(weight * self._yaw_ownership(self.current_layer))
        outgoing_weight = _f32(1.0 - weight)
        outgoing = _f32(outgoing_weight * self._yaw_ownership(self.from_layer))
        return _clamp01(_f32(current + outgoing))

    def get_reference_frame(
        self,
        frames_ahead: int,
        *,
        samples_for_clip: Callable[[NpzClip], ClipSamples],
        num_dofs: int,
        mirror_tables: MirrorTables | None = None,
        quaternion_slerp: QuaternionSlerp | None = None,
        atan2f: Atan2F | None = None,
        sin_cos_f: SinCosF | None = None,
    ) -> PoseSample:
        return get_reference_frame(
            self.current_layer,
            self.from_layer,
            frames_ahead,
            samples_for_clip=samples_for_clip,
            num_dofs=num_dofs,
            w_in=self.w_in,
            w_out=self.w_out,
            xt=self.xt,
            mirror_tables=mirror_tables,
            quaternion_slerp=quaternion_slerp,
            atan2f=atan2f,
            sin_cos_f=sin_cos_f,
        )


@dataclass
class SingleLayerComposer:
    """The recovered non-crossfading portion of ``SonicMotionComposer``."""

    current_layer: Layer = field(default_factory=Layer)
    action_playing: bool = False

    def advance(self) -> AdvanceResult:
        result = advance_layer(self.current_layer)
        if result.completed:
            self.action_playing = False
            callback = self.current_layer.on_complete
            self.current_layer.on_complete = None
            if callback is not None:
                callback()
        return result

    def begin_crossfade(self, *_args: Any, **_kwargs: Any) -> None:
        raise UnsupportedComposerSemantics(
            "crossfade weights and feature-matched entry are not implemented"
        )
