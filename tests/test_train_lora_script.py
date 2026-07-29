import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train_lora.sh"


class TrainLoRAScriptTest(unittest.TestCase):
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

    def test_maps_rank_and_control_configuration_to_dora(self) -> None:
        result = self.run_train(
            "--rank",
            "16",
            "--epochs",
            "1",
            "--updates-per-epoch",
            "10",
            "--warmup-steps",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = set(result.stdout.splitlines())
        expected = {
            "transformer_lm.lora.enabled=true",
            "transformer_lm.lora.rank=16",
            "transformer_lm.lora.alpha=16",
            "transformer_lm.lora.dropout=0.05",
            "transformer_lm.lora.condition_gated=true",
            "distillation.enabled=false",
            "optim.ema.use=false",
            "classifier_free_guidance.training_dropout=0",
            "conditioners.description.t5.word_dropout=0",
            "dataset.train.merge_text_p=0",
            "dataset.train.drop_desc_p=0",
            "dataset.train.drop_other_p=0",
            "optim.lr=1e-4",
            "seed=2036",
        }
        self.assertTrue(expected.issubset(arguments), expected - arguments)

    def test_maps_distillation_configuration_and_defaults_to_dora(self) -> None:
        result = self.run_train(
            "--distill",
            "--epochs",
            "1",
            "--updates-per-epoch",
            "10",
            "--warmup-steps",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = set(result.stdout.splitlines())
        expected = {
            "transformer_lm.lora.enabled=true",
            "transformer_lm.lora.rank=16",
            "transformer_lm.lora.alpha=16",
            "transformer_lm.lora.condition_gated=false",
            "dataset.batch_size=1",
            "optim.lr=3e-5",
            "distillation.enabled=true",
            "distillation.teacher_checkpoint=facebook/musicgen-large",
            "distillation.temperature=2",
            "distillation.kl_weight=0.75",
            "distillation.ce_weight=0.25",
            "distillation.cfg_branches=true",
        }
        self.assertTrue(expected.issubset(arguments), expected - arguments)

    def test_supports_conditional_only_distillation(self) -> None:
        result = self.run_train(
            "--distill",
            "--conditional-only",
            "--epochs",
            "1",
            "--updates-per-epoch",
            "2",
            "--warmup-steps",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "distillation.cfg_branches=false",
            result.stdout.splitlines(),
        )

    def test_rejects_zero_distillation_weight(self) -> None:
        result = self.run_train("--distill", "--kd-weight", "0")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--kd-weight", result.stderr)

    def test_rejects_adapter_dropout_of_one(self) -> None:
        result = self.run_train("--adapter-dropout", "1")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--adapter-dropout", result.stderr)

    def test_help_lists_adapter_options(self) -> None:
        result = subprocess.run(
            ["bash", str(TRAIN_SCRIPT), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--distill", result.stdout)
        self.assertIn("--teacher MODEL", result.stdout)
        self.assertIn("--rank N", result.stdout)
        self.assertIn("--adapter-dropout RATE", result.stdout)


if __name__ == "__main__":
    unittest.main()
