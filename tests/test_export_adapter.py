import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


EXPORTER_PATH = Path(__file__).resolve().parents[1] / "export_adapter.py"
SPEC = importlib.util.spec_from_file_location("export_adapter", EXPORTER_PATH)
assert SPEC is not None and SPEC.loader is not None
export_adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_adapter)


class FakeTensor:
    def __init__(self, size: int) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size


class ExportAdapterTest(unittest.TestCase):
    def checkpoint_package(self) -> dict:
        return {
            "xp.cfg": {
                "transformer_lm": {
                    "lora": {
                        "enabled": True,
                        "rank": 8,
                        "alpha": 8,
                        "dropout": 0.05,
                        "targets": [
                            "self_attention",
                            "cross_attention",
                            "feedforward",
                        ],
                    }
                }
            },
            "best_state": {
                "model": {
                    "transformer.layers.0.self_attn.in_proj_weight": FakeTensor(100),
                    "transformer.layers.0.self_attn.lora_q.lora_a": FakeTensor(80),
                    "transformer.layers.0.self_attn.lora_q.lora_b": FakeTensor(80),
                }
            },
            "fsdp_best_state": {},
        }

    def test_parses_latest_or_positive_epoch(self) -> None:
        self.assertIsNone(export_adapter.parse_checkpoint("latest"))
        self.assertEqual(export_adapter.parse_checkpoint("2"), 2)
        with self.assertRaises(argparse.ArgumentTypeError):
            export_adapter.parse_checkpoint("0")

    def test_exports_only_adapter_tensors_with_provenance(self) -> None:
        checkpoint_package = self.checkpoint_package()
        captured_state = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            checkpoint_path = experiment_dir / "checkpoint_2.th"
            checkpoint_path.write_bytes(b"adapter checkpoint")
            output_dir = root / "adapter"

            train_module = SimpleNamespace(
                main=SimpleNamespace(
                    get_xp_from_sig=lambda _signature: SimpleNamespace(
                        folder=experiment_dir
                    )
                )
            )
            torch_module = ModuleType("torch")
            torch_module.load = lambda *_args, **_kwargs: checkpoint_package

            def save(package: dict, destination: Path) -> None:
                captured_state.append(package)
                Path(destination).write_bytes(b"adapter weights")

            torch_module.save = save
            audiocraft_module = ModuleType("audiocraft")
            audiocraft_module.train = train_module

            with mock.patch.dict(
                sys.modules,
                {
                    "torch": torch_module,
                    "audiocraft": audiocraft_module,
                },
            ):
                export_adapter.export_adapter(
                    "lora-r8",
                    output_dir,
                    checkpoint=2,
                )

            metadata = json.loads(
                (output_dir / "adapter.json").read_text(encoding="utf-8")
            )

        self.assertEqual(metadata["lora"]["rank"], 8)
        self.assertEqual(metadata["trainable_parameters"], 160)
        self.assertEqual(metadata["base_model"], "facebook/musicgen-small")
        self.assertEqual(
            set(captured_state[0]["state_dict"]),
            {
                "transformer.layers.0.self_attn.lora_q.lora_a",
                "transformer.layers.0.self_attn.lora_q.lora_b",
            },
        )

    def test_rejects_non_lora_checkpoint(self) -> None:
        package = self.checkpoint_package()
        package["best_state"]["model"] = {"base.weight": FakeTensor(10)}

        with self.assertRaisesRegex(RuntimeError, "no LoRA adapter tensors"):
            export_adapter.extract_adapter_state(package)


if __name__ == "__main__":
    unittest.main()
