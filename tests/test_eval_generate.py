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
    def test_resolves_pinned_ace_step_model(self) -> None:
        source, record = generate.resolve_model_source(
            generate.ACE_STEP_DEFAULT_MODEL,
            "ace-step",
        )

        self.assertEqual(source, generate.ACE_STEP_DEFAULT_MODEL)
        self.assertEqual(record["backend"], "ace-step")
        self.assertEqual(record["revision"], generate.ACE_STEP_DEFAULT_REVISION)
        self.assertEqual(
            record["library_version"],
            generate.ACE_STEP_PACKAGE_VERSION,
        )
        self.assertEqual(record["model_config"], "acestep-v15-turbo")
        self.assertEqual(record["parameter_scale"], "2B")
        self.assertEqual(record["source_revision"], generate.ACE_STEP_SOURCE_REVISION)

    def test_rejects_other_ace_step_model(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "supports only the 2B Turbo"):
            generate.resolve_model_source(
                "ACE-Step/a-different-model",
                "ace-step",
            )

    def test_reports_uninitialized_ace_step_submodule(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                generate,
                "ACE_STEP_SOURCE_DIR",
                Path(temporary_directory) / "ace-step",
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "git submodule update --init ace-step",
            ),
        ):
            generate.require_ace_step_submodule()

    def test_reports_missing_ace_step_environment_dependency(self) -> None:
        real_import = __import__

        def fail_soundfile_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "soundfile":
                raise ImportError("No module named 'soundfile'", name="soundfile")
            return real_import(name, globals, locals, fromlist, level)

        with (
            mock.patch.object(
                generate,
                "require_ace_step_submodule",
                return_value=Path("/ace-step"),
            ),
            mock.patch("builtins.__import__", side_effect=fail_soundfile_import),
            self.assertRaisesRegex(
                RuntimeError,
                "could not import 'soundfile'.*uv sync --project ace-step",
            ),
        ):
            generate.generate_ace_step(
                SimpleNamespace(),
                Path("/unused"),
                generate.ACE_STEP_DEFAULT_MODEL,
                {},
                [],
            )

    def test_builds_locked_ace_step_config(self) -> None:
        args = SimpleNamespace(
            backend="ace-step",
            run_name="ace-test",
            seeds=[42],
            batch_size=1,
            cfg_coef=3.0,
            adapter_scale=1.0,
            ace_steps=12,
            ace_guidance_scale=1.0,
            ace_shift=2.0,
            ace_cpu_offload=False,
            ace_quantization="int8_weight_only",
        )
        model_source = {
            "type": "pretrained",
            "backend": "ace-step",
            "model_id": generate.ACE_STEP_DEFAULT_MODEL,
        }

        with mock.patch.object(generate, "sha256_file", return_value="digest"):
            config = generate.build_locked_config(
                args,
                [{"id": "prompt-0"}],
                "prompt-digest",
                model_source,
            )

        self.assertEqual(config["schema_version"], 3)
        self.assertEqual(config["backend"], "ace-step")
        self.assertNotIn("audiocraft_commit", config)
        self.assertEqual(config["generation"]["lyrics"], "[Instrumental]")
        self.assertEqual(config["generation"]["inference_steps"], 12)
        self.assertEqual(config["generation"]["shift"], 2.0)
        self.assertTrue(config["generation"]["instrumental"])
        self.assertFalse(config["generation"]["thinking"])
        self.assertFalse(config["generation"]["enable_normalization"])
        self.assertEqual(config["audio_write"]["subtype"], "PCM_16")
        self.assertFalse(config["ace_cpu_offload"])
        self.assertEqual(config["ace_quantization"], "int8_weight_only")

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
            adapter_scale=1.0,
        )

        with self.assertRaisesRegex(
            RuntimeError, "--batch-size must be greater than zero"
        ):
            generate.validate_args(args, prompt_count=1)

    def test_requires_single_clip_batches_for_ace_step(self) -> None:
        args = SimpleNamespace(
            backend="ace-step",
            run_name="test-run",
            limit=None,
            seeds=[42],
            model=generate.ACE_STEP_DEFAULT_MODEL,
            batch_size=2,
            cfg_coef=3.0,
            adapter_scale=1.0,
            ace_steps=8,
            ace_guidance_scale=1.0,
            ace_shift=3.0,
        )

        with self.assertRaisesRegex(RuntimeError, "requires --batch-size 1"):
            generate.validate_args(args, prompt_count=1)

    def test_rejects_invalid_cfg_coefficient(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            limit=None,
            seeds=[42],
            model="model",
            batch_size=1,
            cfg_coef=float("inf"),
            adapter_scale=1.0,
        )

        with self.assertRaisesRegex(
            RuntimeError, "--cfg-coef must be a finite positive number"
        ):
            generate.validate_args(args, prompt_count=1)

    def test_rejects_invalid_duration(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            limit=None,
            seeds=[42],
            model="model",
            batch_size=1,
            duration=0,
            cfg_coef=3.0,
            adapter_scale=1.0,
        )

        with self.assertRaisesRegex(
            RuntimeError, "--duration must be a finite positive number"
        ):
            generate.validate_args(args, prompt_count=1)

    def test_rejects_invalid_adapter_scale(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            limit=None,
            seeds=[42],
            model="model",
            batch_size=1,
            cfg_coef=3.0,
            adapter_scale=-0.1,
        )

        with self.assertRaisesRegex(
            RuntimeError, "--adapter-scale must be a finite non-negative number"
        ):
            generate.validate_args(args, prompt_count=1)

    def test_locks_requested_cfg_coefficient_and_adapter_scale(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            seeds=[42],
            batch_size=1,
            duration=10.0,
            cfg_coef=5.0,
            adapter_scale=0.25,
        )

        with mock.patch.object(generate, "sha256_file", return_value="digest"):
            config = generate.build_locked_config(
                args,
                [{"id": "prompt-0"}],
                "prompt-digest",
                {"type": "lora_adapter", "base_model": "test-model"},
            )

        self.assertEqual(config["generation"]["duration"], 10.0)
        self.assertEqual(config["generation"]["cfg_coef"], 5.0)
        self.assertEqual(config["adapter_scale"], 0.25)
        self.assertEqual(generate.DEFAULT_GENERATION_PARAMS["cfg_coef"], 3.0)

    def test_rejects_adapter_scale_for_non_adapter_model(self) -> None:
        args = SimpleNamespace(
            run_name="test-run",
            seeds=[42],
            batch_size=1,
            cfg_coef=3.0,
            adapter_scale=0.5,
        )

        with self.assertRaisesRegex(
            RuntimeError, "--adapter-scale only applies to LoRA adapter models"
        ):
            generate.build_locked_config(
                args,
                [{"id": "prompt-0"}],
                "prompt-digest",
                {"type": "pretrained", "model_id": "test-model"},
            )

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
            lambda lm, config: calls.append(("inject", lm, dict(config)))
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
                    "lora": {
                        "enabled": True,
                        "rank": 16,
                        "condition_gated": False,
                    },
                },
                "cpu",
            )

        self.assertIs(loaded, model)
        self.assertEqual(calls[0], ("base", "facebook/musicgen-small", "cpu"))
        self.assertEqual(calls[1][0], "inject")
        self.assertEqual(calls[1][2]["rank"], 16)
        self.assertFalse(calls[1][2]["condition_gated"])
        self.assertEqual(calls[2][0], "load")
        self.assertEqual(calls[3], "eval")

    def test_applies_runtime_scale_to_every_lora_projection(self) -> None:
        class FakeProjection:
            def __init__(self, scaling: float) -> None:
                self.scaling = scaling

        projections = [FakeProjection(1.0), FakeProjection(0.5)]
        model = SimpleNamespace(modules=lambda: [model, *projections])
        lora_module = ModuleType("audiocraft.modules.lora")
        lora_module.GatedLoRAProjection = FakeProjection

        with mock.patch.dict(
            sys.modules,
            {"audiocraft.modules.lora": lora_module},
        ):
            count = generate.apply_adapter_scale(model, 0.25)

        self.assertEqual(count, 2)
        self.assertEqual(
            [projection.scaling for projection in projections],
            [0.25, 0.125],
        )

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
            "adapter_scale": 1.0,
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

    def test_generates_ace_step_2b_prompts_with_native_api(self) -> None:
        prompts = [
            {"id": f"prompt-{index}", "cohort": "test", "prompt": f"Prompt {index}"}
            for index in range(2)
        ]
        clip_plan = generate.build_clip_plan(prompts, [42])
        args = SimpleNamespace(
            device="cpu",
            model=generate.ACE_STEP_DEFAULT_MODEL,
            batch_size=1,
            ace_checkpoints_dir=None,
        )
        model_source = {
            "type": "pretrained",
            "backend": "ace-step",
            "model_id": generate.ACE_STEP_DEFAULT_MODEL,
            "revision": generate.ACE_STEP_DEFAULT_REVISION,
            "model_config": generate.ACE_STEP_MODEL_CONFIG,
            "parameter_scale": "2B",
            "library": "ace-step",
            "library_version": generate.ACE_STEP_PACKAGE_VERSION,
            "source_revision": generate.ACE_STEP_SOURCE_REVISION,
        }
        locked_config = {
            "backend": "ace-step",
            "model_source": model_source,
            "adapter_scale": 1.0,
            "ace_cpu_offload": False,
            "ace_quantization": None,
            "generation": {
                **generate.ACE_STEP_DEFAULT_PARAMS,
                "inference_steps": 8,
            },
            "audio_write": {"format": "wav", "subtype": "PCM_16"},
        }
        generation_calls = []
        initialization_calls = []
        snapshot_calls = []
        written = []
        normalized = []

        class FakeAceWaveform:
            ndim = 2
            shape = (2, 16)

            def detach(self) -> "FakeAceWaveform":
                return self

            def cpu(self) -> "FakeAceWaveform":
                return self

            def float(self) -> "FakeAceWaveform":
                return self

            def transpose(self, *_args: int) -> "FakeAceWaveform":
                return self

            def numpy(self) -> list[float]:
                return [0.0]

        class FakeAceStepHandler:
            sample_rate = 48_000
            dtype = "torch.float32"
            quantization = None

            def initialize_service(self, **kwargs: object) -> tuple[str, bool]:
                initialization_calls.append(kwargs)
                self.quantization = kwargs["quantization"]
                return "initialized", True

        class FakeGenerationParams:
            def __init__(self, **kwargs: object) -> None:
                self.values = kwargs

        class FakeGenerationConfig:
            def __init__(self, **kwargs: object) -> None:
                self.values = kwargs

        def fake_generate_music(
            handler: object,
            llm_handler: object,
            params: FakeGenerationParams,
            config: FakeGenerationConfig,
            *,
            save_dir: object,
        ) -> object:
            generation_calls.append(
                (handler, llm_handler, params.values, config.values, save_dir)
            )
            return SimpleNamespace(
                success=True,
                error=None,
                status_message="ok",
                audios=[
                    {
                        "tensor": FakeAceWaveform(),
                        "sample_rate": 48_000,
                    }
                ],
            )

        def fake_snapshot_download(**kwargs: object) -> str:
            snapshot_calls.append(kwargs)
            return str(kwargs["local_dir"])

        seeded = []
        torch_module = ModuleType("torch")
        torch_module.__version__ = "test"
        torch_module.version = SimpleNamespace(cuda=None)
        torch_module.cuda = SimpleNamespace(is_available=lambda: False)
        torch_module.manual_seed = seeded.append
        torch_module.inference_mode = nullcontext

        soundfile_module = ModuleType("soundfile")

        def soundfile_write(path: str, *_args: object, **kwargs: object) -> None:
            Path(path).write_bytes(b"ace audio")
            written.append((path, kwargs))

        soundfile_module.write = soundfile_write
        torchaudio_module = ModuleType("torchaudio")
        torchaudio_module.__version__ = "test"
        acestep_module = ModuleType("acestep")
        acestep_handler_module = ModuleType("acestep.handler")
        acestep_handler_module.AceStepHandler = FakeAceStepHandler
        acestep_inference_module = ModuleType("acestep.inference")
        acestep_inference_module.GenerationConfig = FakeGenerationConfig
        acestep_inference_module.GenerationParams = FakeGenerationParams
        acestep_inference_module.generate_music = fake_generate_music
        huggingface_hub_module = ModuleType("huggingface_hub")
        huggingface_hub_module.snapshot_download = fake_snapshot_download
        modules = {
            "torch": torch_module,
            "soundfile": soundfile_module,
            "torchaudio": torchaudio_module,
            "acestep": acestep_module,
            "acestep.handler": acestep_handler_module,
            "acestep.inference": acestep_inference_module,
            "huggingface_hub": huggingface_hub_module,
        }
        versions = {
            "ace-step": generate.ACE_STEP_PACKAGE_VERSION,
            "diffusers": "test",
            "transformers": "test",
            "huggingface-hub": "test",
            "soundfile": "test",
        }

        def normalize_audio(
            _torch: object,
            _torchaudio: object,
            waveform: object,
            sample_rate: int,
        ) -> object:
            normalized.append(sample_rate)
            return waveform

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "run"
            checkpoints_dir = Path(temporary_directory) / "checkpoints"
            args.ace_checkpoints_dir = str(checkpoints_dir)
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.dict("os.environ", {}, clear=False),
                mock.patch.object(
                    generate,
                    "package_version",
                    side_effect=lambda name: versions.get(name),
                ),
                mock.patch.object(
                    generate,
                    "require_ace_step_submodule",
                    return_value=Path("/ace-step-source"),
                ),
                mock.patch.object(
                    generate,
                    "verify_ace_step_source_checkout",
                    return_value=(
                        Path("/ace-step-source"),
                        generate.ACE_STEP_SOURCE_REVISION,
                    ),
                ),
                mock.patch.object(
                    generate,
                    "normalize_ace_step_audio",
                    side_effect=normalize_audio,
                ),
                redirect_stdout(io.StringIO()),
            ):
                generate.generate(
                    args,
                    output_dir,
                    generate.ACE_STEP_DEFAULT_MODEL,
                    locked_config,
                    clip_plan,
                )

            records = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(
            snapshot_calls[0]["revision"],
            generate.ACE_STEP_DEFAULT_REVISION,
        )
        self.assertEqual(snapshot_calls[0]["repo_id"], generate.ACE_STEP_DEFAULT_MODEL)
        self.assertEqual(
            initialization_calls[0]["config_path"],
            generate.ACE_STEP_MODEL_CONFIG,
        )
        self.assertFalse(initialization_calls[0]["offload_to_cpu"])
        self.assertIsNone(initialization_calls[0]["quantization"])
        self.assertEqual(
            [call[2]["caption"] for call in generation_calls],
            ["Prompt 0", "Prompt 1"],
        )
        self.assertEqual(generation_calls[0][2]["lyrics"], "[Instrumental]")
        self.assertTrue(generation_calls[0][2]["instrumental"])
        self.assertEqual(generation_calls[0][2]["duration"], 30)
        self.assertEqual(generation_calls[0][2]["inference_steps"], 8)
        self.assertFalse(generation_calls[0][2]["thinking"])
        self.assertFalse(generation_calls[0][2]["enable_normalization"])
        self.assertEqual(generation_calls[0][3]["seeds"], [42])
        self.assertFalse(generation_calls[0][3]["use_random_seed"])
        self.assertEqual(seeded, [42, 42])
        self.assertEqual(len(written), 2)
        self.assertEqual(normalized, [48_000, 48_000])
        self.assertEqual(records[0]["backend"], "ace-step")
        self.assertNotIn("audiocraft_commit", records[0])
        self.assertEqual(records[0]["sample_rate"], 48_000)


if __name__ == "__main__":
    unittest.main()
