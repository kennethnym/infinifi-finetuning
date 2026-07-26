import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train.sh"


class TrainScriptTest(unittest.TestCase):
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

    def test_maps_conditioning_and_cadence_options_to_dora(self) -> None:
        result = self.run_train(
            "--lr",
            "2e-6",
            "--epochs",
            "3",
            "--updates-per-epoch",
            "500",
            "--warmup-steps",
            "100",
            "--generate-every",
            "1",
            "--checkpoint-every",
            "1",
            "--word-dropout",
            "0.1",
            "--cfg-dropout",
            ".1",
            "--merge-text-p",
            "0",
            "--drop-desc-p",
            "0.0",
            "--drop-other-p",
            "0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        expected = {
            "optim.lr=2e-6",
            "optim.epochs=3",
            "optim.updates_per_epoch=500",
            "schedule.cosine.warmup=100",
            "generate.every=1",
            "checkpoint.save_every=1",
            "conditioners.description.t5.word_dropout=0.1",
            "classifier_free_guidance.training_dropout=.1",
            "dataset.train.merge_text_p=0",
            "dataset.train.drop_desc_p=0.0",
            "dataset.train.drop_other_p=0",
        }
        self.assertTrue(expected.issubset(arguments), expected - set(arguments))

    def test_omits_optional_conditioning_overrides_by_default(self) -> None:
        result = self.run_train("--epochs", "1", "--updates-per-epoch", "1")

        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertIn("generate.every=5", arguments)
        self.assertIn("checkpoint.save_every=5", arguments)
        self.assertFalse(
            any("word_dropout=" in argument for argument in arguments)
        )
        self.assertFalse(
            any(
                argument.startswith("classifier_free_guidance.training_dropout=")
                for argument in arguments
            )
        )

    def test_rejects_probability_outside_unit_interval(self) -> None:
        result = self.run_train("--word-dropout", "1.1")

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--word-dropout must be a number between 0 and 1",
            result.stderr,
        )

    def test_help_lists_new_options(self) -> None:
        result = subprocess.run(
            ["bash", str(TRAIN_SCRIPT), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--word-dropout RATE", result.stdout)
        self.assertIn("--cfg-dropout RATE", result.stdout)
        self.assertIn("--checkpoint-every N", result.stdout)


if __name__ == "__main__":
    unittest.main()
