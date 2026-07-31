import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train_scratch.sh"


class TrainScratchScriptTest(unittest.TestCase):
    def run_train(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            fake_dora = bin_dir / "dora"
            fake_dora.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\"\n",
                encoding="utf-8",
            )
            fake_dora.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
            return subprocess.run(
                ["bash", str(TRAIN_SCRIPT), *arguments],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_defaults_build_random_compact_student_distillation(self) -> None:
        result = self.run_train()

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = set(result.stdout.splitlines())
        expected = {
            "model/lm/model_scale=lofi_student",
            "continue_from=null",
            "transformer_lm.dim=640",
            "transformer_lm.num_heads=10",
            "transformer_lm.num_layers=10",
            "transformer_lm.lora.enabled=false",
            "conditioners.description.t5.finetune=false",
            "dataset.batch_size=1",
            "dataset.segment_duration=10",
            "optim.epochs=20",
            "optim.updates_per_epoch=1000",
            "optim.grad_accumulation_steps=8",
            "optim.ema.use=false",
            "distillation.enabled=true",
            "distillation.teacher_checkpoint=facebook/musicgen-large",
            "distillation.temperature=2",
            "distillation.initial_kl_weight=0.5",
            "distillation.initial_ce_weight=0.5",
            "distillation.kl_weight=0.75",
            "distillation.ce_weight=0.25",
            "distillation.weight_schedule_updates=10000",
            "distillation.cfg_branches=true",
        }
        self.assertTrue(expected.issubset(arguments), expected - arguments)
        self.assertFalse(any("musicgen-small" in item for item in arguments))

    def test_maps_resume_architecture_and_manifest_overrides(self) -> None:
        result = self.run_train(
            "--continue-from",
            "//sig/student-stage-one",
            "--dim",
            "640",
            "--heads",
            "10",
            "--layers",
            "8",
            "--train-data",
            "/data/overfit",
            "--valid-data",
            "/data/valid",
            "--conditional-only",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertIn("continue_from=//sig/student-stage-one", arguments)
        self.assertNotIn("continue_from=null", arguments)
        self.assertIn("transformer_lm.dim=640", arguments)
        self.assertIn("transformer_lm.num_heads=10", arguments)
        self.assertIn("transformer_lm.num_layers=8", arguments)
        self.assertIn("datasource.train=/data/overfit", arguments)
        self.assertIn("datasource.valid=/data/valid", arguments)
        self.assertIn("datasource.evaluate=/data/valid", arguments)
        self.assertIn("distillation.cfg_branches=false", arguments)

    def test_rejects_incompatible_attention_width(self) -> None:
        result = self.run_train("--dim", "640", "--heads", "12")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--dim must be divisible by --heads", result.stderr)

    def test_rejects_zero_weight_pairs_and_accumulation(self) -> None:
        initial = self.run_train(
            "--initial-kd-weight", "0", "--initial-ce-weight", "0"
        )
        final = self.run_train("--kd-weight", "0", "--ce-weight", "0")
        accumulation = self.run_train("--grad-accumulation", "0")

        self.assertEqual(initial.returncode, 2)
        self.assertIn("initial CE and KD weights", initial.stderr)
        self.assertEqual(final.returncode, 2)
        self.assertIn("final CE and KD weights", final.stderr)
        self.assertEqual(accumulation.returncode, 2)
        self.assertIn("--grad-accumulation", accumulation.stderr)

    def test_help_describes_scratch_specific_controls(self) -> None:
        result = subprocess.run(
            ["bash", str(TRAIN_SCRIPT), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--weight-transition-updates", result.stdout)
        self.assertIn("--grad-accumulation", result.stdout)
        self.assertIn("--continue-from", result.stdout)


if __name__ == "__main__":
    unittest.main()
