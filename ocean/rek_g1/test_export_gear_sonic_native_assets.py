import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import export_gear_sonic_native_assets as exporter


class ExportFailureCleanupTests(unittest.TestCase):
    def test_partial_export_directory_is_removed(self) -> None:
        motion = SimpleNamespace(
            dof_pos=np.zeros((2, 29), dtype=np.float32),
            root_pos=np.zeros((2, 3), dtype=np.float32),
            root_rot_xyzw=np.asarray(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            role="idle",
            filename="idle.npz",
            size=1,
            sha256="motion",
            fps=50.0,
            manifest_sha256="manifest",
            inventory_sha256="inventory",
        )
        xml_contract = SimpleNamespace(source_bytes=b"<mujoco/>", source_sha256="xml")
        arena_contract = SimpleNamespace(
            source_sha256="arena", derived_geometry_sha256="geometry"
        )
        real_write = exporter._write_new
        calls = 0

        def fail_second_write(path: Path, payload: bytes) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected export failure")
            return real_write(path, payload)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "native-assets"
            arguments = [
                "export_gear_sonic_native_assets.py",
                "--assets-dir",
                str(Path(temporary) / "assets"),
                "--motion-role",
                "idle",
                "--manifest",
                str(Path(temporary) / "manifest.json"),
                "--xml",
                str(Path(temporary) / "model.xml"),
                "--arena",
                str(Path(temporary) / "arena.json"),
                "--out",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(exporter.plant, "load_motion", return_value=motion),
                mock.patch.object(
                    exporter.plant, "inspect_xml_contract", return_value=xml_contract
                ),
                mock.patch.object(
                    exporter.plant, "load_arena_contract", return_value=arena_contract
                ),
                mock.patch.object(
                    exporter.plant, "add_arena_geoms", return_value="<mujoco/>"
                ),
                mock.patch.object(exporter, "_write_new", side_effect=fail_second_write),
            ):
                with self.assertRaisesRegex(OSError, "injected export failure"):
                    exporter.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
