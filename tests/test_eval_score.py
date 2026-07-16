import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCORER_PATH = Path(__file__).resolve().parents[1] / "eval" / "score.py"
SPEC = importlib.util.spec_from_file_location("eval_score", SCORER_PATH)
assert SPEC is not None and SPEC.loader is not None
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


class EvalScoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.original_runs_root = score.RUNS_ROOT
        self.original_prompts_path = score.PROMPTS_PATH
        self.original_prompts_checksum_path = score.PROMPTS_CHECKSUM_PATH
        score.RUNS_ROOT = self.root / "runs"
        score.PROMPTS_PATH = self.root / "prompts.jsonl"
        score.PROMPTS_CHECKSUM_PATH = self.root / "prompts.sha256"
        score.PROMPTS_PATH.write_text(
            json.dumps(
                {
                    "id": "dataset-1",
                    "cohort": "dataset_eval",
                    "source_id": "source-1",
                    "paired_id": None,
                    "prompt": "Quiet piano lo-fi",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        score.PROMPTS_CHECKSUM_PATH.write_text(
            score.sha256_file(score.PROMPTS_PATH) + "  prompts.jsonl\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        score.RUNS_ROOT = self.original_runs_root
        score.PROMPTS_PATH = self.original_prompts_path
        score.PROMPTS_CHECKSUM_PATH = self.original_prompts_checksum_path
        self.temporary_directory.cleanup()

    def make_run(self) -> Path:
        run_dir = score.RUNS_ROOT / "test-run"
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "dataset-1__seed-42.wav").write_bytes(b"generated")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_name": "test-run",
                    "model_source": {
                        "type": "pretrained",
                        "model_id": "facebook/musicgen-small",
                    },
                    "audiocraft_commit": score.AUDIOCRAFT_COMMIT,
                    "prompt_manifest_sha256": score.sha256_file(score.PROMPTS_PATH),
                    "prompt_ids": ["dataset-1"],
                    "seeds": [42],
                    "generation": {"duration": 30},
                }
            ),
            encoding="utf-8",
        )
        manifest_record = {
            "clip_id": "dataset-1__seed-42",
            "prompt_id": "dataset-1",
            "cohort": "dataset_eval",
            "source_id": "source-1",
            "paired_id": None,
            "prompt": "Quiet piano lo-fi",
            "seed": 42,
            "audio_path": "audio/dataset-1__seed-42.wav",
            "duration_seconds": 30,
            "sample_rate": 32_000,
            "model_source": {
                "type": "pretrained",
                "model_id": "facebook/musicgen-small",
            },
            "audiocraft_commit": score.AUDIOCRAFT_COMMIT,
        }
        (run_dir / "manifest.jsonl").write_text(
            json.dumps(manifest_record) + "\n",
            encoding="utf-8",
        )
        return run_dir

    def test_loads_complete_generated_run(self) -> None:
        run_dir = self.make_run()

        loaded_dir, config, records = score.load_run("test-run")

        self.assertEqual(loaded_dir, run_dir.resolve())
        self.assertEqual(config["seeds"], [42])
        self.assertEqual(records[0]["source_id"], "source-1")

    def test_rejects_incomplete_generated_run(self) -> None:
        run_dir = self.make_run()
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["seeds"].append(43)
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "incomplete or inconsistent"):
            score.load_run("test-run")

    def test_rejects_manifest_identity_mismatch(self) -> None:
        run_dir = self.make_run()
        manifest_path = run_dir / "manifest.jsonl"
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["clip_id"] = "wrong-id"
        manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "clip_id does not match"):
            score.load_run("test-run")

    def test_rejects_frozen_prompt_mismatch(self) -> None:
        run_dir = self.make_run()
        manifest_path = run_dir / "manifest.jsonl"
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["prompt"] = "Different prompt"
        manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Frozen prompt field prompt"):
            score.load_run("test-run")

    def test_resolves_reference_from_metadata(self) -> None:
        self.make_run()
        _, _, records = score.load_run("test-run")
        reference_dir = self.root / "references"
        reference_dir.mkdir()
        (reference_dir / "track.json").write_text(
            json.dumps(
                {
                    "name": "source-1",
                    "description": "Quiet piano lo-fi",
                }
            ),
            encoding="utf-8",
        )
        (reference_dir / "track.mp3").write_bytes(b"reference")

        references = score.load_references(reference_dir, records)

        self.assertEqual(set(references), {"source-1"})
        self.assertEqual(
            references["source-1"]["sha256"],
            score.sha256_file(reference_dir / "track.mp3"),
        )

    def test_weighted_summary_uses_segment_counts(self) -> None:
        result = score.summary([1.0, 4.0], [1, 3])

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["weight"], 4)
        self.assertEqual(result["mean"], 3.25)
        self.assertAlmostEqual(result["std"], 1.299038105676658)

    def test_score_lock_is_exclusive(self) -> None:
        run_dir = self.make_run()
        first_lock = score.acquire_score_lock(run_dir)
        try:
            with self.assertRaisesRegex(RuntimeError, "Another scorer"):
                score.acquire_score_lock(run_dir)
        finally:
            first_lock.close()

        second_lock = score.acquire_score_lock(run_dir)
        second_lock.close()

    def test_existing_result_requires_matching_output_hashes(self) -> None:
        run_dir = self.make_run()
        score_config = {"schema_version": 1, "run_name": "test-run"}
        clip_metrics_path = run_dir / "clip_metrics.jsonl"
        metrics_path = run_dir / "metrics.json"
        clip_metrics_path.write_text("{}\n", encoding="utf-8")
        metrics_path.write_text("{}\n", encoding="utf-8")
        score.write_json(
            run_dir / "score_config.json",
            {
                **score_config,
                "outputs": {
                    "clip_metrics_sha256": score.sha256_file(clip_metrics_path),
                    "metrics_sha256": score.sha256_file(metrics_path),
                },
            },
        )

        self.assertTrue(
            score.existing_result_action(run_dir, score_config, overwrite=False)
        )
        metrics_path.write_text('{"changed": true}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "already exists or is incomplete"):
            score.existing_result_action(run_dir, score_config, overwrite=False)


if __name__ == "__main__":
    unittest.main()
