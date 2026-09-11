#!/usr/bin/env python3
"""Bind serialized T800 policy absence to the observed Windows client lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from controller_path import BUILD_FINGERPRINT, validate_report


SCHEMA = "rek.t800_client_runtime_boundary.v1"
MISSING_PROFILE = re.compile(
    r"^\[EngineAIPolicyRunner\] Profile '([^']+)' missing onnxBytes or configJson\.$"
)
VISUAL_ONLY = re.compile(
    r"^\[Robot\] Visual-only mode: disabled (\d+) MuJoCo components on "
    r"(engineai_t800_[^(]+)\(Clone\); colliders preserved for local VFX\.$"
)
NETWORK_CLIENT = re.compile(
    r"^\[Robot\.Network\] Initialized on (engineai_t800_[^(]+)\(Clone\): "
    r"(\d+) bones, role=Client$"
)


class EvidenceError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not a JSON object")
    return value


def _all_payloads_null(profiles: Any) -> bool:
    if not isinstance(profiles, list) or len(profiles) != 45:
        return False
    null_pointer = {"m_FileID": 0, "m_PathID": 0}
    return all(
        isinstance(profile, dict)
        and all(profile.get(field) == null_pointer
                for field in ("onnxBytes", "configJson", "trajectoryCsv"))
        for profile in profiles
    )


def build_report(
    player_log_path: Path | str,
    controller_path: Path | str,
    authority_path: Path | str,
) -> dict[str, Any]:
    player_log_path = Path(player_log_path)
    controller_path = Path(controller_path)
    authority_path = Path(authority_path)
    controller = _load_object(controller_path, "controller path")
    controller_errors = validate_report(controller)
    if controller_errors:
        raise EvidenceError(
            "controller path validation failed: " + "; ".join(controller_errors[:4])
        )
    authority = _load_object(authority_path, "authority artifact")
    authority_verdict = (authority.get("verdict") or {}).get("verdict")
    authority_fingerprint = (authority.get("scope") or {}).get("build_fingerprint")
    if authority_verdict != "remote_authority":
        raise EvidenceError("authority artifact does not establish remote authority")
    if authority_fingerprint != BUILD_FINGERPRINT:
        raise EvidenceError("authority artifact build fingerprint mismatch")
    if controller.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise EvidenceError("controller artifact build fingerprint mismatch")

    runner = ((controller.get("serialized_t800") or {}).get("runner") or {})
    profiles = (runner.get("values") or {}).get("profiles")
    if not _all_payloads_null(profiles):
        raise EvidenceError("controller artifact does not preserve 45 null T800 payload sets")
    profile_names = {profile.get("name") for profile in profiles}

    lines = player_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    missing: list[dict[str, Any]] = []
    visual: list[dict[str, Any]] = []
    network: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = MISSING_PROFILE.fullmatch(line)
        if match:
            profile = match.group(1)
            if profile not in profile_names:
                raise EvidenceError(
                    f"runtime missing-profile name {profile!r} is absent from serialized runner"
                )
            stack_window = lines[index + 1:index + 9]
            if not any("REKApp.EngineAIPolicyRunner:LoadProfile(" in item
                       for item in stack_window):
                raise EvidenceError("missing-profile message has no LoadProfile stack frame")
            if not any(item == "REKApp.EngineAIPolicyRunner:Init()"
                       for item in stack_window):
                raise EvidenceError("missing-profile message has no Init stack frame")
            missing.append({
                "line": index + 1,
                "profile": profile,
                "load_profile_stack_observed": True,
                "init_stack_observed": True,
            })
            continue
        match = VISUAL_ONLY.fullmatch(line)
        if match:
            visual.append({
                "line": index + 1,
                "disabled_mujoco_components": int(match.group(1)),
                "robot": match.group(2),
            })
            continue
        match = NETWORK_CLIENT.fullmatch(line)
        if match:
            network.append({
                "line": index + 1,
                "robot": match.group(1),
                "bones": int(match.group(2)),
                "role": "Client",
            })

    if not missing:
        raise EvidenceError("Player.log has no T800 missing-profile initialization event")
    if not visual:
        raise EvidenceError("Player.log has no T800 visual-only transition")
    if not network:
        raise EvidenceError("Player.log has no T800 network-client initialization")
    if any(item["disabled_mujoco_components"] != 148 for item in visual):
        raise EvidenceError("T800 visual-only component count drifted")
    if any(item["bones"] != 26 for item in network):
        raise EvidenceError("T800 network skeleton count drifted")

    return {
        "schema": SCHEMA,
        "build_fingerprint": BUILD_FINGERPRINT,
        "sources": {
            "player_log": {
                "path": str(player_log_path.resolve()),
                "sha256": _sha256(player_log_path),
                "disclosure": "Only strict allowlisted facts are copied; no session or account data is emitted.",
            },
            "controller_path": {
                "path": str(controller_path.resolve()),
                "sha256": _sha256(controller_path),
                "schema": controller.get("schema"),
            },
            "authority": {
                "path": str(authority_path.resolve()),
                "sha256": _sha256(authority_path),
                "verdict": authority_verdict,
            },
        },
        "serialized_client": {
            "runner_count_in_bound_probe": 1,
            "profiles_in_factory_runner": len(profiles),
            "profile_payload_sets_all_null": True,
            "payload_fields": ["onnxBytes", "configJson", "trajectoryCsv"],
        },
        "runtime_observations": {
            "missing_profile_init_failures": missing,
            "visual_only_transitions": visual,
            "network_client_initializations": network,
        },
        "verdict": {
            "client_t800_runner_state": "initialization_aborted_on_missing_profile_payload",
            "client_t800_physics_state": "disabled_in_observed_remote_authority_mode",
            "client_t800_render_state": "network_driven_visual_replica",
            "client_contains_authoritative_t800_policy_payload": False,
            "authoritative_controller_payload_location": "unknown",
        },
        "limits": [
            "The result applies to the pinned Windows client and observed remote-authority mode.",
            "It does not prove which server build, weights, configuration, or hidden state produced the poses.",
            "Absent client payloads cannot be recovered by additional client file scanning.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-log", required=True, type=Path)
    parser.add_argument("--controller-path", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    report = build_report(args.player_log, args.controller_path, args.authority)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "missing_profile_init_failures": len(
            report["runtime_observations"]["missing_profile_init_failures"]
        ),
        "visual_only_transitions": len(
            report["runtime_observations"]["visual_only_transitions"]
        ),
        "network_client_initializations": len(
            report["runtime_observations"]["network_client_initializations"]
        ),
        "client_contains_authoritative_t800_policy_payload": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
