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
        )

        with self.assertRaisesRegex(
            RuntimeError, "--batch-size must be greater than zero"
        ):
            generate.validate_args(args, prompt_count=1)

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
        self.assertEqual(resumed_calls, [["Prompt 0", "Prompt 1"]])
        self.assertEqual(seeded, [42])
        self.assertEqual(len(resumed_records), 6)


if __name__ == "__main__":
    unittest.main()
