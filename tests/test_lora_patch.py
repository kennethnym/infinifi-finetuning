from pathlib import Path
import re
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIOCRAFT_COMMIT = "adf0b04a4452f171970028fcf80f101dd5e26e19"


class LoRAPatchIntegrationTest(unittest.TestCase):
    def test_patch_and_runtime_use_the_pinned_submodule_commit(self) -> None:
        tree = subprocess.run(
            ["git", "ls-tree", "HEAD", "audiocraft"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn(AUDIOCRAFT_COMMIT, tree)

        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(f"ARG AUDIOCRAFT_COMMIT={AUDIOCRAFT_COMMIT}", dockerfile)
        self.assertIn(
            "apply --check /workspace/patches/audiocraft-lora.patch",
            dockerfile,
        )
        self.assertIn(
            "apply /workspace/patches/audiocraft-lora.patch",
            dockerfile,
        )
        self.assertIn(
            "apply --check /workspace/patches/audiocraft-scratch.patch",
            dockerfile,
        )
        self.assertIn(
            "apply /workspace/patches/audiocraft-scratch.patch",
            dockerfile,
        )

        for relative_path in (
            "export_adapter.py",
            "eval/generate.py",
        ):
            contents = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            match = re.search(r'^AUDIOCRAFT_COMMIT = "([0-9a-f]{40})"$', contents, re.M)
            self.assertIsNotNone(match, relative_path)
            self.assertEqual(match.group(1), AUDIOCRAFT_COMMIT)

    def test_patch_contains_native_lora_and_regression_tests(self) -> None:
        patch = (
            PROJECT_ROOT / "patches" / "audiocraft-lora.patch"
        ).read_text(encoding="utf-8")
        expected_headers = {
            "diff --git a/audiocraft/modules/lora.py b/audiocraft/modules/lora.py",
            "diff --git a/audiocraft/modules/transformer.py b/audiocraft/modules/transformer.py",
            "diff --git a/audiocraft/models/lm.py b/audiocraft/models/lm.py",
            "diff --git a/audiocraft/solvers/musicgen.py b/audiocraft/solvers/musicgen.py",
            "diff --git a/config/solver/musicgen/default.yaml b/config/solver/musicgen/default.yaml",
            "diff --git a/tests/modules/test_lora.py b/tests/modules/test_lora.py",
            "diff --git a/tests/models/test_lm_lora.py b/tests/models/test_lm_lora.py",
            "diff --git a/tests/solvers/test_musicgen_distillation.py b/tests/solvers/test_musicgen_distillation.py",
        }
        self.assertTrue(
            expected_headers.issubset(set(patch.splitlines())),
            expected_headers - set(patch.splitlines()),
        )

    def test_scratch_patch_contains_full_model_distillation_support(self) -> None:
        patch = (
            PROJECT_ROOT / "patches" / "audiocraft-scratch.patch"
        ).read_text(encoding="utf-8")
        expected_headers = {
            "diff --git a/audiocraft/solvers/base.py b/audiocraft/solvers/base.py",
            "diff --git a/audiocraft/solvers/builders.py b/audiocraft/solvers/builders.py",
            "diff --git a/audiocraft/solvers/musicgen.py b/audiocraft/solvers/musicgen.py",
            "diff --git a/config/model/lm/model_scale/lofi_student.yaml b/config/model/lm/model_scale/lofi_student.yaml",
            "diff --git a/config/solver/musicgen/default.yaml b/config/solver/musicgen/default.yaml",
        }
        self.assertTrue(
            expected_headers.issubset(set(patch.splitlines())),
            expected_headers - set(patch.splitlines()),
        )
        self.assertIn("_scheduled_distillation_weights", patch)
        self.assertIn("grad_accumulation_steps", patch)
        self.assertIn("self.compression_model.requires_grad_(False)", patch)


if __name__ == "__main__":
    unittest.main()
