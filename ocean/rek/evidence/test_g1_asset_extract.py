import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_asset_extract as extractor


def make_npz(role: str, extra_member: bool = False) -> tuple[bytes, list[dict]]:
    contents = {
        name: f"{role}:{name}".encode("ascii")
        for name in sorted(extractor.EXPECTED_ARCHIVE_MEMBERS)
    }
    if extra_member:
        contents["unexpected.npy"] = b"unexpected"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    members = [
        {
            "name": name,
            "bytes": len(content),
            "sha256": extractor.sha256_bytes(content),
        }
        for name, content in contents.items()
        if name in extractor.EXPECTED_ARCHIVE_MEMBERS
    ]
    return output.getvalue(), members


def synthetic_manifest(extra_member_role: str | None = None):
    source = b"synthetic sharedassets0 fixture"
    assets = []
    payloads = {}
    for path_id, role in enumerate(sorted(extractor.EXPECTED_ROLES), start=100):
        payload, members = make_npz(role, extra_member=role == extra_member_role)
        name = role + "_processed"
        asset = {
            "role": role,
            "path_id": path_id,
            "name": name,
            "output": name + ".npz",
            "bytes": len(payload),
            "sha256": extractor.sha256_bytes(payload),
            "frames": path_id,
            "dof": 29,
            "fps": 50.0,
            "members": members,
        }
        if role in extractor.MOVE_INDEX_BY_ROLE:
            move_index = extractor.MOVE_INDEX_BY_ROLE[role]
            asset["robot_config_move_index"] = move_index
            asset["mocap_clip_config_path_id"] = (
                extractor.MOVE_CONFIG_PATH_ID_BY_INDEX[move_index]
            )
        assets.append(asset)
        payloads[path_id] = payload
    manifest = {
        "schema": extractor.MANIFEST_SCHEMA,
        "build_fingerprint": "1" * 64,
        "parser": {"name": "UnityPy", "version": "1.25.2"},
        "source": {
            "name": "sharedassets0.assets",
            "bytes": len(source),
            "sha256": extractor.sha256_bytes(source),
        },
        "robot_config": {
            "robot_id": extractor.ROBOT_ID,
            "path_id": extractor.ROBOT_CONFIG_PATH_ID,
            "name": extractor.ROBOT_CONFIG_NAME,
            "serialized_sha256": extractor.ROBOT_CONFIG_SHA256,
            "move_count": len(extractor.MOVE_BINDINGS),
        },
        "assets": assets,
    }
    return source, manifest, payloads


def loaded_records(manifest, payloads):
    return [
        extractor.LoadedTextAsset(
            path_id=item["path_id"],
            type_name="TextAsset",
            name=item["name"],
            payload=payloads[item["path_id"]],
        )
        for item in manifest["assets"]
    ]


class PinnedManifestTests(unittest.TestCase):
    def test_pinned_manifest_matches_tracked_evidence(self):
        raw = extractor.MANIFEST_PATH.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            extractor.manifest_sha256(data), extractor.PINNED_MANIFEST_SHA256
        )
        manifest = extractor.parse_manifest(data)

        schema = json.loads(
            (HERE / "evidence_out" / "processed_textasset_schema_v3.json").read_text(
                encoding="utf-8"
            )
        )
        evidence_by_id = {item["path_id"]: item for item in schema["assets"]}
        self.assertEqual(len(manifest.assets), 21)
        for spec in manifest.assets:
            evidence = evidence_by_id[spec.path_id]
            self.assertEqual(evidence["container"], manifest.source_name)
            self.assertEqual(evidence["name"], spec.name)
            self.assertEqual(evidence["bytes"], spec.size)
            self.assertEqual(evidence["sha256"], spec.sha256)
            self.assertEqual(evidence["measured_fps"], spec.fps)
            dof = next(
                member for member in evidence["members"] if member["name"] == "dof_pos.npy"
            )
            self.assertEqual(dof["shape"], [spec.frames, spec.dof])
            evidence_members = {
                member["name"]: (member["bytes"], member["sha256"])
                for member in evidence["members"]
            }
            spec_members = {
                member.name: (member.size, member.sha256) for member in spec.members
            }
            self.assertEqual(evidence_members, spec_members)

        probe = json.loads(
            (HERE / "evidence_out" / "mujoco_asset_probe_v8.json").read_text(
                encoding="utf-8"
            )
        )
        robot_configs = [
            target
            for target in probe["targets"]
            if target["container"] == manifest.source_name
            and target["class"] == "RobotConfig"
            and target["path_id"] == manifest.robot_config_path_id
        ]
        self.assertEqual(len(robot_configs), 1)
        robot_config = robot_configs[0]
        self.assertEqual(robot_config["serialized_sha256"], manifest.robot_config_sha256)
        self.assertEqual(robot_config["values"]["robotId"], manifest.robot_id)
        self.assertEqual(robot_config["values"]["m_Name"], manifest.robot_config_name)
        move_bindings = robot_config["values"]["moves"]
        self.assertEqual(len(move_bindings), manifest.robot_config_move_count)
        mocap_by_id = {
            target["path_id"]: target
            for target in probe["targets"]
            if target["container"] == manifest.source_name
            and target["class"] == "MocapClipConfig"
        }
        move_specs = [
            spec
            for spec in manifest.assets
            if spec.robot_config_move_index is not None
        ]
        self.assertEqual(
            sorted(spec.robot_config_move_index for spec in move_specs),
            list(range(17)),
        )
        for spec in move_specs:
            move_index = spec.robot_config_move_index
            self.assertIsNotNone(move_index)
            config_path_id = spec.mocap_clip_config_path_id
            self.assertIsNotNone(config_path_id)
            self.assertEqual(
                move_bindings[move_index]["m_PathID"],
                config_path_id,
            )
            mocap = mocap_by_id[config_path_id]
            self.assertEqual(mocap["values"]["m_Name"], spec.name)
            self.assertEqual(mocap["values"]["npzFile"]["m_PathID"], spec.path_id)

        inventory = json.loads(
            (HERE / "evidence_out" / "inventory.json").read_text(encoding="utf-8")
        )
        source = next(
            item
            for item in inventory["files"]
            if item["path"].replace("\\", "/").endswith("/sharedassets0.assets")
        )
        self.assertEqual(inventory["build_fingerprint"], manifest.build_fingerprint)
        self.assertEqual(source["size"], manifest.source_size)
        self.assertEqual(source["sha256"], manifest.source_sha256)

    def test_manifest_requires_exact_roles_and_unique_identities(self):
        _, data, _ = synthetic_manifest()
        duplicate = copy.deepcopy(data)
        duplicate["assets"][1]["path_id"] = duplicate["assets"][0]["path_id"]
        with self.assertRaisesRegex(extractor.ExtractionError, "duplicate path IDs"):
            extractor.parse_manifest(duplicate)

        missing = copy.deepcopy(data)
        missing["assets"].pop()
        with self.assertRaisesRegex(extractor.ExtractionError, "roles must be exactly"):
            extractor.parse_manifest(missing)

        wrong_config = copy.deepcopy(data)
        move = next(
            item
            for item in wrong_config["assets"]
            if item.get("robot_config_move_index") == 0
        )
        move["mocap_clip_config_path_id"] += 1
        with self.assertRaisesRegex(extractor.ExtractionError, "MocapClipConfig mismatch"):
            extractor.parse_manifest(wrong_config)

        wrong_robot = copy.deepcopy(data)
        wrong_robot["robot_config"]["robot_id"] = "unknown"
        with self.assertRaisesRegex(extractor.ExtractionError, "identity mismatch"):
            extractor.parse_manifest(wrong_robot)


class PayloadValidationTests(unittest.TestCase):
    def test_valid_payloads_publish_byte_exact_deterministic_inventory(self):
        source, data, payloads = synthetic_manifest()
        manifest = extractor.parse_manifest(data)
        validated = extractor.validate_loaded_assets(
            manifest, loaded_records(data, payloads)
        )
        inventory_a = extractor.build_inventory(
            manifest,
            "2" * 64,
            len(source),
            extractor.sha256_bytes(source),
        )
        inventory_b = extractor.build_inventory(
            manifest,
            "2" * 64,
            len(source),
            extractor.sha256_bytes(source),
        )
        self.assertEqual(inventory_a, inventory_b)
        inventory_data = json.loads(inventory_a)
        self.assertEqual(
            inventory_data["robot_config"]["move_count"],
            len(extractor.MOVE_BINDINGS),
        )
        inventory_by_role = {
            item["role"]: item for item in inventory_data["assets"]
        }
        self.assertIsNone(inventory_by_role["idle"]["robot_config_move_index"])
        self.assertEqual(
            inventory_by_role["left_hook"]["robot_config_move_index"], 0
        )
        self.assertEqual(
            inventory_by_role["left_hook"]["mocap_clip_config_path_id"], 2704
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "g1-assets"
            inventory_path = extractor._publish_into_new_directory(
                output, validated, inventory_a
            )
            self.assertEqual(inventory_path.parent, output)
            self.assertEqual(inventory_path.read_bytes(), inventory_a)
            for asset in validated:
                self.assertEqual(
                    (output / asset.spec.output).read_bytes(), asset.payload
                )

    def test_name_type_missing_and_payload_hash_mismatches_fail_closed(self):
        _, data, payloads = synthetic_manifest()
        manifest = extractor.parse_manifest(data)

        records = loaded_records(data, payloads)
        records[0] = extractor.LoadedTextAsset(
            records[0].path_id, "TextAsset", "wrong_name", records[0].payload
        )
        with self.assertRaisesRegex(extractor.ExtractionError, "name mismatch"):
            extractor.validate_loaded_assets(manifest, records)

        records = loaded_records(data, payloads)
        records[0] = extractor.LoadedTextAsset(
            records[0].path_id, "AnimationClip", records[0].name, records[0].payload
        )
        with self.assertRaisesRegex(extractor.ExtractionError, "type mismatch"):
            extractor.validate_loaded_assets(manifest, records)

        with self.assertRaisesRegex(extractor.ExtractionError, "missing required"):
            extractor.validate_loaded_assets(manifest, loaded_records(data, payloads)[1:])

        records = loaded_records(data, payloads)
        records[0] = extractor.LoadedTextAsset(
            records[0].path_id,
            records[0].type_name,
            records[0].name,
            records[0].payload + b"corrupt",
        )
        with self.assertRaisesRegex(extractor.ExtractionError, "byte count mismatch"):
            extractor.validate_loaded_assets(manifest, records)

    def test_npz_with_unexpected_member_is_rejected(self):
        _, data, payloads = synthetic_manifest(extra_member_role="walk")
        manifest = extractor.parse_manifest(data)
        with self.assertRaisesRegex(extractor.ExtractionError, "archive members mismatch"):
            extractor.validate_loaded_assets(manifest, loaded_records(data, payloads))

    def test_surrogateescape_round_trips_binary_textasset(self):
        payload = bytes(range(256))
        value = payload.decode("utf-8", "surrogateescape")
        self.assertEqual(extractor._script_bytes(value), payload)


class SourceAndOutputSafetyTests(unittest.TestCase):
    def test_source_name_size_and_hash_are_all_required(self):
        source, data, _ = synthetic_manifest()
        manifest = extractor.parse_manifest(data)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sharedassets0.assets"
            path.write_bytes(source)
            self.assertEqual(
                extractor.validate_source(path, manifest),
                (len(source), extractor.sha256_bytes(source)),
            )
            path.write_bytes(source + b"x")
            with self.assertRaisesRegex(extractor.ExtractionError, "byte count mismatch"):
                extractor.validate_source(path, manifest)

    def test_publication_refuses_git_worktree_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            (repository / ".git").mkdir(parents=True)
            with self.assertRaisesRegex(extractor.ExtractionError, "outside every Git"):
                extractor.reject_unsafe_output(repository / "assets")

            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "sentinel"
            sentinel.write_bytes(b"keep")
            with self.assertRaisesRegex(extractor.ExtractionError, "already exists"):
                extractor.reject_unsafe_output(existing)
            self.assertEqual(sentinel.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
