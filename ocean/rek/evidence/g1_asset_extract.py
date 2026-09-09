"""Extract the build-pinned G1 motion TextAssets without modifying REK.

The command accepts exactly one installed ``sharedassets0.assets`` file and a
caller-selected output directory.  It hashes the complete source container,
loads only the allowlisted TextAsset path IDs, validates names, payload hashes,
and NPZ member hashes, then publishes the original payload bytes.  Validation
finishes before any extracted payload is written.

The output directory must not already exist and must not be inside a Git work
tree.  This keeps proprietary runtime payloads out of this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


MANIFEST_PATH = Path(__file__).with_name("g1_runtime_assets.v1.json")
PINNED_MANIFEST_SHA256 = (
    "06aa831f9ee6a094660df90085a03c2d8ef331137f74d6bb18849716e29c9eae"
)
MANIFEST_SCHEMA = "rek.g1_runtime_assets.manifest.v1"
INVENTORY_SCHEMA = "rek.g1_runtime_assets.inventory.v1"
EXPECTED_ROLES = frozenset(
    {
        "idle",
        "kick_left_front",
        "kick_left_side",
        "kick_right_knee",
        "kick_right_side",
        "strafe_left",
        "turn_right",
        "walk",
    }
)
EXPECTED_ARCHIVE_MEMBERS = frozenset(
    {"dof_pos.npy", "fps.npy", "root_pos.npy", "root_rot.npy"}
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_OUTPUT_RE = re.compile(r"[a-z0-9_]+\.npz\Z")


class ExtractionError(RuntimeError):
    """A fail-closed source, manifest, payload, or publication error."""


@dataclass(frozen=True)
class MemberSpec:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class AssetSpec:
    role: str
    path_id: int
    name: str
    output: str
    size: int
    sha256: str
    frames: int
    dof: int
    fps: float
    members: tuple[MemberSpec, ...]


@dataclass(frozen=True)
class Manifest:
    build_fingerprint: str
    parser: str
    parser_version: str
    source_name: str
    source_size: int
    source_sha256: str
    assets: tuple[AssetSpec, ...]


@dataclass(frozen=True)
class LoadedTextAsset:
    path_id: int
    type_name: str
    name: str
    payload: bytes


@dataclass(frozen=True)
class ValidatedAsset:
    spec: AssetSpec
    payload: bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(data: object) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canonical)


def _require_dict(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ExtractionError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ExtractionError(f"{label} must be an array")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExtractionError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExtractionError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtractionError(f"{label} must be numeric")
    result = float(value)
    if not (result > 0.0):
        raise ExtractionError(f"{label} must be positive")
    return result


def _require_sha256(value: object, label: str) -> str:
    result = _require_str(value, label)
    if SHA256_RE.fullmatch(result) is None:
        raise ExtractionError(f"{label} must be a lowercase SHA-256")
    return result


def parse_manifest(data: object) -> Manifest:
    root = _require_dict(data, "manifest")
    if root.get("schema") != MANIFEST_SCHEMA:
        raise ExtractionError("manifest schema mismatch")

    build_fingerprint = _require_sha256(
        root.get("build_fingerprint"), "build_fingerprint"
    )
    parser = _require_dict(root.get("parser"), "parser")
    parser_name = _require_str(parser.get("name"), "parser.name")
    parser_version = _require_str(parser.get("version"), "parser.version")
    if parser_name != "UnityPy":
        raise ExtractionError("only the pinned UnityPy parser is supported")

    source = _require_dict(root.get("source"), "source")
    source_name = _require_str(source.get("name"), "source.name")
    if source_name != "sharedassets0.assets":
        raise ExtractionError("source.name must be sharedassets0.assets")
    source_size = _require_int(source.get("bytes"), "source.bytes", 1)
    source_sha256 = _require_sha256(source.get("sha256"), "source.sha256")

    specs: list[AssetSpec] = []
    for index, raw_asset in enumerate(_require_list(root.get("assets"), "assets")):
        item = _require_dict(raw_asset, f"assets[{index}]")
        role = _require_str(item.get("role"), f"assets[{index}].role")
        path_id = _require_int(item.get("path_id"), f"assets[{index}].path_id", 1)
        name = _require_str(item.get("name"), f"assets[{index}].name")
        output = _require_str(item.get("output"), f"assets[{index}].output")
        if SAFE_OUTPUT_RE.fullmatch(output) is None or Path(output).name != output:
            raise ExtractionError(f"assets[{index}].output is not a safe NPZ filename")
        size = _require_int(item.get("bytes"), f"assets[{index}].bytes", 1)
        digest = _require_sha256(item.get("sha256"), f"assets[{index}].sha256")
        frames = _require_int(item.get("frames"), f"assets[{index}].frames", 1)
        dof = _require_int(item.get("dof"), f"assets[{index}].dof", 1)
        fps = _require_float(item.get("fps"), f"assets[{index}].fps")

        members: list[MemberSpec] = []
        for member_index, raw_member in enumerate(
            _require_list(item.get("members"), f"assets[{index}].members")
        ):
            member = _require_dict(
                raw_member, f"assets[{index}].members[{member_index}]"
            )
            members.append(
                MemberSpec(
                    name=_require_str(
                        member.get("name"),
                        f"assets[{index}].members[{member_index}].name",
                    ),
                    size=_require_int(
                        member.get("bytes"),
                        f"assets[{index}].members[{member_index}].bytes",
                        1,
                    ),
                    sha256=_require_sha256(
                        member.get("sha256"),
                        f"assets[{index}].members[{member_index}].sha256",
                    ),
                )
            )
        member_names = [member.name for member in members]
        if len(member_names) != len(set(member_names)):
            raise ExtractionError(f"assets[{index}] has duplicate archive members")
        if set(member_names) != EXPECTED_ARCHIVE_MEMBERS:
            raise ExtractionError(
                f"assets[{index}] must pin exactly {sorted(EXPECTED_ARCHIVE_MEMBERS)}"
            )
        specs.append(
            AssetSpec(
                role=role,
                path_id=path_id,
                name=name,
                output=output,
                size=size,
                sha256=digest,
                frames=frames,
                dof=dof,
                fps=fps,
                members=tuple(sorted(members, key=lambda value: value.name)),
            )
        )

    roles = [spec.role for spec in specs]
    path_ids = [spec.path_id for spec in specs]
    names = [spec.name for spec in specs]
    outputs = [spec.output for spec in specs]
    if set(roles) != EXPECTED_ROLES or len(roles) != len(EXPECTED_ROLES):
        raise ExtractionError(f"manifest roles must be exactly {sorted(EXPECTED_ROLES)}")
    for label, values in (
        ("path IDs", path_ids),
        ("asset names", names),
        ("output names", outputs),
    ):
        if len(values) != len(set(values)):
            raise ExtractionError(f"manifest has duplicate {label}")
    if any(spec.dof != 29 for spec in specs):
        raise ExtractionError("every pinned G1 motion must contain 29 DoF")
    if any(spec.fps != 50.0 for spec in specs):
        raise ExtractionError("every pinned G1 motion must be sampled at 50 Hz")

    return Manifest(
        build_fingerprint=build_fingerprint,
        parser=parser_name,
        parser_version=parser_version,
        source_name=source_name,
        source_size=source_size,
        source_sha256=source_sha256,
        assets=tuple(sorted(specs, key=lambda value: value.path_id)),
    )


def load_pinned_manifest() -> tuple[Manifest, str]:
    raw = MANIFEST_PATH.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"pinned manifest is invalid JSON: {exc}") from exc
    digest = manifest_sha256(data)
    if digest != PINNED_MANIFEST_SHA256:
        raise ExtractionError(
            "pinned manifest hash mismatch; review and repin the source manifest"
        )
    return parse_manifest(data), digest


def validate_source(path: Path, manifest: Manifest) -> tuple[int, str]:
    if path.name != manifest.source_name:
        raise ExtractionError(
            f"source filename must be exactly {manifest.source_name!r}"
        )
    if path.is_symlink() or not path.is_file():
        raise ExtractionError("source must be a regular, non-symlink file")
    size = path.stat().st_size
    if size != manifest.source_size:
        raise ExtractionError(
            f"source byte count mismatch: expected {manifest.source_size}, got {size}"
        )
    digest = sha256_file(path)
    if digest != manifest.source_sha256:
        raise ExtractionError(
            f"source SHA-256 mismatch: expected {manifest.source_sha256}, got {digest}"
        )
    return size, digest


def _script_bytes(value: object) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8", "surrogateescape")
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise ExtractionError(f"TextAsset m_Script has unsupported type {type(value).__name__}")


def load_unity_text_assets(path: Path, manifest: Manifest) -> list[LoadedTextAsset]:
    try:
        import UnityPy
    except ImportError as exc:
        raise ExtractionError(
            f"UnityPy {manifest.parser_version} is required"
        ) from exc
    version = str(getattr(UnityPy, "__version__", ""))
    if version != manifest.parser_version:
        raise ExtractionError(
            f"UnityPy version mismatch: expected {manifest.parser_version}, got {version!r}"
        )

    expected_ids = {spec.path_id for spec in manifest.assets}
    try:
        environment = UnityPy.load(str(path))
    except Exception as exc:
        raise ExtractionError(f"UnityPy could not load the source container: {exc}") from exc

    loaded: list[LoadedTextAsset] = []
    seen_ids: set[int] = set()
    for obj in environment.objects:
        try:
            path_id = int(obj.path_id)
        except (AttributeError, TypeError, ValueError):
            continue
        if path_id not in expected_ids:
            continue
        if path_id in seen_ids:
            raise ExtractionError(f"duplicate Unity object path ID {path_id}")
        seen_ids.add(path_id)
        type_name = str(getattr(getattr(obj, "type", None), "name", ""))
        if type_name != "TextAsset":
            raise ExtractionError(
                f"path ID {path_id} type mismatch: expected TextAsset, got {type_name!r}"
            )
        try:
            parser = getattr(obj, "parse_as_object", None)
            data = parser() if callable(parser) else obj.read()
            name = str(getattr(data, "m_Name"))
            payload = _script_bytes(getattr(data, "m_Script"))
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"could not read Unity object path ID {path_id}: {exc}") from exc
        loaded.append(
            LoadedTextAsset(
                path_id=path_id,
                type_name=type_name,
                name=name,
                payload=payload,
            )
        )
    return loaded


def validate_npz(payload: bytes, spec: AssetSpec) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ExtractionError(f"{spec.name} contains duplicate archive members")
            expected = {member.name: member for member in spec.members}
            if set(names) != set(expected):
                raise ExtractionError(
                    f"{spec.name} archive members mismatch: expected {sorted(expected)}, got {sorted(names)}"
                )
            for info in infos:
                member = expected[info.filename]
                if info.flag_bits & 0x1:
                    raise ExtractionError(f"{spec.name}/{info.filename} is encrypted")
                if info.is_dir() or Path(info.filename).name != info.filename:
                    raise ExtractionError(f"{spec.name}/{info.filename} is not a flat file")
                if info.file_size != member.size:
                    raise ExtractionError(
                        f"{spec.name}/{info.filename} byte count mismatch"
                    )
                content = archive.read(info)
                if len(content) != member.size or sha256_bytes(content) != member.sha256:
                    raise ExtractionError(f"{spec.name}/{info.filename} SHA-256 mismatch")
    except ExtractionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ExtractionError(f"{spec.name} is not the pinned NPZ payload: {exc}") from exc


def validate_loaded_assets(
    manifest: Manifest, loaded: Iterable[LoadedTextAsset]
) -> tuple[ValidatedAsset, ...]:
    by_id: dict[int, LoadedTextAsset] = {}
    for record in loaded:
        if record.path_id in by_id:
            raise ExtractionError(f"duplicate loaded path ID {record.path_id}")
        by_id[record.path_id] = record

    expected_ids = {spec.path_id for spec in manifest.assets}
    unexpected = sorted(set(by_id) - expected_ids)
    if unexpected:
        raise ExtractionError(f"loader returned unexpected path IDs: {unexpected}")
    missing = sorted(expected_ids - set(by_id))
    if missing:
        raise ExtractionError(f"missing required TextAsset path IDs: {missing}")

    result: list[ValidatedAsset] = []
    for spec in manifest.assets:
        record = by_id[spec.path_id]
        if record.type_name != "TextAsset":
            raise ExtractionError(
                f"path ID {spec.path_id} type mismatch: expected TextAsset, got {record.type_name!r}"
            )
        if record.name != spec.name:
            raise ExtractionError(
                f"path ID {spec.path_id} name mismatch: expected {spec.name!r}, got {record.name!r}"
            )
        if len(record.payload) != spec.size:
            raise ExtractionError(f"{spec.name} payload byte count mismatch")
        if sha256_bytes(record.payload) != spec.sha256:
            raise ExtractionError(f"{spec.name} payload SHA-256 mismatch")
        validate_npz(record.payload, spec)
        result.append(ValidatedAsset(spec=spec, payload=record.payload))
    return tuple(result)


def build_inventory(
    manifest: Manifest,
    manifest_sha256: str,
    source_size: int,
    source_sha256: str,
) -> bytes:
    report = {
        "schema": INVENTORY_SCHEMA,
        "build_fingerprint": manifest.build_fingerprint,
        "manifest_sha256": manifest_sha256,
        "source": {
            "name": manifest.source_name,
            "bytes": source_size,
            "sha256": source_sha256,
        },
        "assets": [
            {
                "role": spec.role,
                "path_id": spec.path_id,
                "name": spec.name,
                "output": spec.output,
                "bytes": spec.size,
                "sha256": spec.sha256,
                "frames": spec.frames,
                "dof": spec.dof,
                "fps": spec.fps,
                "members": [
                    {
                        "name": member.name,
                        "bytes": member.size,
                        "sha256": member.sha256,
                    }
                    for member in spec.members
                ],
            }
            for spec in manifest.assets
        ],
    }
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def reject_unsafe_output(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if os.path.lexists(absolute):
        raise ExtractionError("output directory already exists; refusing to overwrite it")
    for component in (absolute, *absolute.parents):
        if component.exists() and _is_link_or_junction(component):
            raise ExtractionError(f"output path traverses a link or junction: {component}")
        if (component / ".git").exists():
            raise ExtractionError("output directory must be outside every Git work tree")
    if absolute.name in ("", ".", ".."):
        raise ExtractionError("output directory is invalid")
    return absolute


def _write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _publish_into_new_directory(
    output_dir: Path,
    assets: Sequence[ValidatedAsset],
    inventory: bytes,
) -> Path:
    output = Path(os.path.abspath(output_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise ExtractionError("output directory already exists; refusing to overwrite it")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for asset in assets:
            _write_exclusive(stage / asset.spec.output, asset.payload)
        inventory_path = stage / "g1_runtime_assets.inventory.json"
        _write_exclusive(inventory_path, inventory)
        if os.path.lexists(output):
            raise ExtractionError("output directory appeared during publication")
        os.rename(stage, output)
        return output / inventory_path.name
    except Exception:
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(
            f".{output.name}.stage-"
        ):
            shutil.rmtree(stage)
        raise


def publish(
    output_dir: Path,
    assets: Sequence[ValidatedAsset],
    inventory: bytes,
) -> Path:
    output = reject_unsafe_output(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = reject_unsafe_output(output)
    return _publish_into_new_directory(output, assets, inventory)


Loader = Callable[[Path, Manifest], list[LoadedTextAsset]]


def extract(
    source: Path,
    output_dir: Path,
    *,
    loader: Loader = load_unity_text_assets,
) -> Path:
    manifest, manifest_sha256 = load_pinned_manifest()
    source_size, source_sha256 = validate_source(source, manifest)
    loaded = loader(source, manifest)
    validated = validate_loaded_assets(manifest, loaded)
    inventory = build_inventory(
        manifest, manifest_sha256, source_size, source_sha256
    )
    return publish(output_dir, validated, inventory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        required=True,
        type=Path,
        help="installed sharedassets0.assets from the pinned REK build",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="new local directory outside any Git work tree",
    )
    arguments = parser.parse_args(argv)
    try:
        inventory = extract(arguments.assets, arguments.out_dir)
    except (ExtractionError, OSError) as exc:
        print(f"g1 asset extraction failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "asset_count": len(EXPECTED_ROLES),
                "inventory": str(inventory),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
