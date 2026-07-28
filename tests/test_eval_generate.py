from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


GENERATOR_PATH = Path(__file__).resolve().parents[1] / "eval" / "generate.py"
SPEC = importlib.util.spec_from_file_location("eval_generate", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
generate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate)


class FakeWaveform:
    def cpu(self) -> "FakeWaveform":
        return self


class FakeModel:
    sample_rate = 32_000

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.generation_params = None

    def set_generation_params(self, **params: object) -> None:
        self.generation_params = params

    def generate(
        self, prompts: list[str], *, progress: bool
    ) -> list[FakeWaveform]:
        assert progress
        self.calls.append(prompts)
        return [FakeWaveform() for _ in prompts]


class EvalGenerateTest(unittest.TestCase):
    def test_builds_seed_homogeneous_batches(self) -> None:
        prompts = [
            {"id": f"prompt-{index}", "cohort": "test", "prompt": f"Prompt {index}"}
            for index in range(3)
        ]
        clip_plan = generate.build_clip_plan(prompts, [42, 43])

        batches = generate.build_generation_batches(clip_plan, 2)

        self.assertEqual(
            [[clip["prompt_id"] for clip in batch] for batch in batches],
            [
                ["prompt-0", "prompt-1"],
                ["prompt-2"],
                ["prompt-0", "prompt-1"],
                ["prompt-2"],
            ],
        )
        self.assertEqual(
            [[clip["seed"] for clip in batch] for batch in batches],
            [[42, 42], [42], [43, 43], [43]],
        )

    def test_rejects_non_positive_batch_size(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            limit=None,
            seeds=[42],
            model="model",
            batch_size=0,
            cfg_coef=3.0,
        )

        with self.assertRaisesRegex(
            RuntimeError, "--batch-size must be greater than zero"
        ):
            generate.validate_args(args, prompt_count=1)

    def test_rejects_invalid_cfg_coefficient(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            limit=None,
            seeds=[42],
            model="model",
            batch_size=1,
            cfg_coef=float("inf"),
        )

        with self.assertRaisesRegex(
            RuntimeError, "--cfg-coef must be a finite positive number"
        ):
            generate.validate_args(args, prompt_count=1)

    def test_locks_requested_cfg_coefficient(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            seeds=[42],
            batch_size=1,
            cfg_coef=5.0,
        )

        with mock.patch.object(generate, "sha256_file", return_value="digest"):
            config = generate.build_locked_config(
                args,
                [{"id": "prompt-0"}],
                "prompt-digest",
                {"type": "pretrained", "model_id": "test-model"},
            )

        self.assertEqual(config["generation"]["cfg_coef"], 5.0)
        self.assertEqual(generate.DEFAULT_GENERATION_PARAMS["cfg_coef"], 3.0)

    def test_resolves_adapter_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter_dir = Path(temporary_directory)
            weights_path = adapter_dir / generate.ADAPTER_WEIGHTS
            weights_path.write_bytes(b"adapter weights")
            metadata = {
                "schema_version": 1,
                "format": "infinifi_musicgen_lora",
                "base_model": "facebook/musicgen-small",
                "audiocraft_commit": generate.AUDIOCRAFT_COMMIT,
                "audiocraft_patch_sha256": generate.sha256_file(
                    generate.AUDIOCRAFT_PATCH
                ),
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
                },
                "files": {
                    generate.ADAPTER_WEIGHTS: {
                        "sha256": generate.sha256_file(weights_path),
                        "size_bytes": weights_path.stat().st_size,
                    }
                },
            }
            (adapter_dir / generate.ADAPTER_METADATA).write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            source, record = generate.resolve_model_source(str(adapter_dir))

        self.assertEqual(source, str(adapter_dir.resolve()))
        self.assertEqual(record["type"], "lora_adapter")
        self.assertEqual(record["base_model"], "facebook/musicgen-small")
        self.assertEqual(record["lora"]["rank"], 8)

    def test_loads_adapter_on_the_frozen_base_model(self) -> None:
        calls = []

        class FakeLM:
            def eval(self) -> None:
                calls.append("eval")

        model = SimpleNamespace(lm=FakeLM())

        class FakeMusicGen:
            @staticmethod
            def get_pretrained(source: str, *, device: str) -> object:
                calls.append(("base", source, device))
                return model

        torch_module = SimpleNamespace(
            load=lambda *_args, **_kwargs: {
                "format": "infinifi_musicgen_lora",
                "state_dict": {"layer.lora_a": "tensor"},
            }
        )
        audiocraft_module = ModuleType("audiocraft")
        modules_module = ModuleType("audiocraft.modules")
        lora_module = ModuleType("audiocraft.modules.lora")
        lora_module.inject_lora = (
            lambda lm, config: calls.append(("inject", lm, config["rank"]))
        )
        lora_module.load_adapter_state_dict = (
            lambda lm, state: calls.append(("load", lm, state))
        )

        with mock.patch.dict(
            sys.modules,
            {
                "audiocraft": audiocraft_module,
                "audiocraft.modules": modules_module,
                "audiocraft.modules.lora": lora_module,
            },
        ):
            loaded = generate.load_musicgen_model(
                FakeMusicGen,
                torch_module,
                "/adapter",
                {
                    "type": "lora_adapter",
                    "base_model": "facebook/musicgen-small",
                    "lora": {"enabled": True, "rank": 8},
                },
                "cpu",
            )

        self.assertIs(loaded, model)
        self.assertEqual(calls[0], ("base", "facebook/musicgen-small", "cpu"))
        self.assertEqual(calls[1][0], "inject")
        self.assertEqual(calls[2][0], "load")
        self.assertEqual(calls[3], "eval")

    def test_generates_prompts_in_batches(self) -> None:
        prompts = [
            {"id": f"prompt-{index}", "cohort": "test", "prompt": f"Prompt {index}"}
            for index in range(3)
        ]
        clip_plan = generate.build_clip_plan(prompts, [42, 43])
        args = SimpleNamespace(
            device="cpu",
            model="test-model",
            batch_size=2,
        )
        locked_config = {
            "model_source": {"type": "pretrained", "model_id": "test-model"},
            "generation": generate.generation_params(4.0),
        }
        fake_model = FakeModel()
        seeded = []

        torch_module = ModuleType("torch")
        torch_module.__version__ = "test"
        torch_module.version = SimpleNamespace(cuda=None)
        torch_module.cuda = SimpleNamespace(is_available=lambda: False)
        torch_module.manual_seed = seeded.append
        torch_module.inference_mode = nullcontext

        audiocraft_module = ModuleType("audiocraft")
        data_module = ModuleType("audiocraft.data")
        audio_module = ModuleType("audiocraft.data.audio")
        models_module = ModuleType("audiocraft.models")
        musicgen_module = ModuleType("audiocraft.models.musicgen")

        def audio_write(path: str, *_args: object, **_kwargs: object) -> None:
            Path(f"{path}.wav").write_bytes(b"audio")

        class FakeMusicGen:
            @staticmethod
            def get_pretrained(_source: str, *, device: str) -> FakeModel:
                self.assertEqual(device, "cpu")
                return fake_model

        audio_module.audio_write = audio_write
        musicgen_module.MusicGen = FakeMusicGen
        modules = {
            "torch": torch_module,
            "audiocraft": audiocraft_module,
            "audiocraft.data": data_module,
            "audiocraft.data.audio": audio_module,
            "audiocraft.models": models_module,
            "audiocraft.models.musicgen": musicgen_module,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "run"
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(generate, "package_version", return_value="test"),
                redirect_stdout(io.StringIO()),
            ):
                generate.generate(
                    args,
                    output_dir,
                    "test-model",
                    locked_config,
                    clip_plan,
                )

            records = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            initial_calls = list(fake_model.calls)
            initial_seeded = list(seeded)

            missing_record = records.pop(1)
            (output_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (output_dir / missing_record["audio_path"]).unlink()
            fake_model.calls.clear()
            seeded.clear()

            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(generate, "package_version", return_value="test"),
                redirect_stdout(io.StringIO()),
            ):
                generate.generate(
                    args,
                    output_dir,
                    "test-model",
                    locked_config,
                    clip_plan,
                )

            resumed_calls = list(fake_model.calls)
            resumed_records = (output_dir / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(
            initial_calls,
            [
                ["Prompt 0", "Prompt 1"],
                ["Prompt 2"],
                ["Prompt 0", "Prompt 1"],
                ["Prompt 2"],
            ],
        )
        self.assertEqual(initial_seeded, [42, 42, 43, 43])
        self.assertEqual(fake_model.generation_params["cfg_coef"], 4.0)
        self.assertEqual(resumed_calls, [["Prompt 0", "Prompt 1"]])
        self.assertEqual(seeded, [42])
        self.assertEqual(len(resumed_records), 6)


if __name__ == "__main__":
    unittest.main()
