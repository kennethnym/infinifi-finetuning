import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


EXPORTER_PATH = Path(__file__).resolve().parents[1] / "export_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("export_checkpoint", EXPORTER_PATH)
assert SPEC is not None and SPEC.loader is not None
export_checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_checkpoint)


class ExportCheckpointTest(unittest.TestCase):
    def test_parses_latest_or_positive_epoch(self) -> None:
        self.assertIsNone(export_checkpoint.parse_checkpoint("latest"))
        self.assertEqual(export_checkpoint.parse_checkpoint("2"), 2)

        for invalid in ("0", "-1", "best", "1/../2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    export_checkpoint.parse_checkpoint(invalid)

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "checkpoint must be a positive epoch number",
            ):
                export_checkpoint.export_checkpoint(
                    "1c050b6d",
                    Path(temporary_directory) / "export",
                    checkpoint=0,
                )

    def test_exports_selected_epoch_with_provenance(self) -> None:
        captured_checkpoint = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            checkpoint_path = experiment_dir / "checkpoint_2.th"
            checkpoint_path.write_bytes(b"epoch two checkpoint")
            output_dir = root / "export"

            train_module = SimpleNamespace(
                main=SimpleNamespace(
                    get_xp_from_sig=lambda signature: SimpleNamespace(
                        folder=experiment_dir,
                    )
                )
            )

            def export_lm(source: Path, destination: str) -> None:
                captured_checkpoint.append(source)
                Path(destination).write_bytes(b"language model")

            def export_compression(_model: str, destination: str) -> None:
                Path(destination).write_bytes(b"compression model")

            export_module = SimpleNamespace(
                export_lm=export_lm,
                export_pretrained_compression_model=export_compression,
            )
            audiocraft_module = ModuleType("audiocraft")
            audiocraft_module.train = train_module
            utils_module = ModuleType("audiocraft.utils")
            utils_module.export = export_module

            with mock.patch.dict(
                sys.modules,
                {
                    "audiocraft": audiocraft_module,
                    "audiocraft.utils": utils_module,
                },
            ):
                export_checkpoint.export_checkpoint(
                    "1c050b6d",
                    output_dir,
                    checkpoint=2,
                )

            metadata = json.loads(
                (output_dir / "export.json").read_text(encoding="utf-8")
            )

            self.assertEqual(captured_checkpoint, [checkpoint_path])
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(
                metadata["source_checkpoint"],
                {
                    "filename": "checkpoint_2.th",
                    "sha256": export_checkpoint.sha256_file(checkpoint_path),
                },
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "source checkpoint does not match",
            ):
                export_checkpoint.export_checkpoint(
                    "1c050b6d",
                    output_dir,
                    checkpoint=1,
                )

    def test_accepts_legacy_latest_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for filename in export_checkpoint.MODEL_FILES:
                (output_dir / filename).write_bytes(filename.encode("utf-8"))
            metadata = {
                "schema_version": 1,
                "signature": "1c050b6d",
                "compression_model": export_checkpoint.COMPRESSION_MODEL,
                "audiocraft_commit": export_checkpoint.AUDIOCRAFT_COMMIT,
                "files": {
                    filename: export_checkpoint.file_metadata(output_dir / filename)
                    for filename in export_checkpoint.MODEL_FILES
                },
            }
            (output_dir / export_checkpoint.EXPORT_METADATA).write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            compatible, reason = export_checkpoint.validate_existing_export(
                output_dir,
                "1c050b6d",
                "checkpoint.th",
            )

        self.assertTrue(compatible)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
