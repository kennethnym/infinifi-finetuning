from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


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

    def write_fad_manifest(
        self,
        corpus_dir: Path,
        records: list[dict[str, object]],
    ) -> None:
        manifest_path = corpus_dir / "manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        attribution_path = corpus_dir / "ATTRIBUTION.md"
        (corpus_dir / "manifest.sha256").write_text(
            (
                f"{score.sha256_file(manifest_path)}  manifest.jsonl\n"
                f"{score.sha256_file(attribution_path)}  ATTRIBUTION.md\n"
            ),
            encoding="utf-8",
        )

    def make_fad_corpus(
        self,
        reference_set: str = "human-fma-lofi-v1",
        *,
        directory_name: str | None = None,
        count: int = 2,
    ) -> tuple[Path, list[dict[str, object]]]:
        corpus_dir = self.root / (directory_name or reference_set)
        audio_dir = corpus_dir / "audio"
        audio_dir.mkdir(parents=True)
        (corpus_dir / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "reference_set": reference_set,
                    "target_count": count,
                }
            ),
            encoding="utf-8",
        )
        (corpus_dir / "ATTRIBUTION.md").write_text(
            f"# {reference_set}\n",
            encoding="utf-8",
        )
        records: list[dict[str, object]] = []
        for index in range(count):
            audio_path = audio_dir / f"track-{index}.wav"
            audio_path.write_bytes(f"reference-{index}".encode())
            records.append(
                {
                    "schema_version": 1,
                    "reference_id": f"track-{index}",
                    "reference_set": reference_set,
                    "audio_path": f"audio/track-{index}.wav",
                    "audio_sha256": score.sha256_file(audio_path),
                }
            )
        self.write_fad_manifest(corpus_dir, records)
        return corpus_dir, records

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

    def test_rejects_duplicate_generation_seeds(self) -> None:
        run_dir = self.make_run()
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["seeds"].append(42)
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "invalid seeds"):
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

    def test_clap_seed_diversity_pairs_seeds_within_each_prompt(self) -> None:
        records = [
            {
                "clip_id": "prompt-a__seed-42",
                "prompt_id": "prompt-a",
                "cohort": "dataset_eval",
                "prompt": "Prompt A",
                "seed": 42,
            },
            {
                "clip_id": "prompt-a__seed-43",
                "prompt_id": "prompt-a",
                "cohort": "dataset_eval",
                "prompt": "Prompt A",
                "seed": 43,
            },
            {
                "clip_id": "prompt-b__seed-42",
                "prompt_id": "prompt-b",
                "cohort": "dataset_eval",
                "prompt": "Prompt B",
                "seed": 42,
            },
            {
                "clip_id": "prompt-b__seed-43",
                "prompt_id": "prompt-b",
                "cohort": "dataset_eval",
                "prompt": "Prompt B",
                "seed": 43,
            },
            {
                "clip_id": "prompt-b__seed-44",
                "prompt_id": "prompt-b",
                "cohort": "dataset_eval",
                "prompt": "Prompt B",
                "seed": 44,
            },
        ]
        embeddings = {
            "prompt-a__seed-42": [1.0, 0.0],
            "prompt-a__seed-43": [0.0, 1.0],
            "prompt-b__seed-42": [2.0, 0.0],
            "prompt-b__seed-43": [1.0, 0.0],
            "prompt-b__seed-44": [3.0, 0.0],
        }

        result = score.summarize_clap_seed_diversity(records, embeddings)

        self.assertEqual(set(result["by_prompt"]), {"prompt-a", "prompt-b"})
        self.assertEqual(result["by_prompt"]["prompt-a"]["prompt"], "Prompt A")
        self.assertEqual(result["by_prompt"]["prompt-a"]["pair_count"], 1)
        self.assertEqual(result["by_prompt"]["prompt-a"]["mean"], 1.0)
        self.assertEqual(result["by_prompt"]["prompt-b"]["pair_count"], 3)
        self.assertEqual(result["by_prompt"]["prompt-b"]["mean"], 0.0)
        self.assertEqual(result["overall"]["count"], 2)
        self.assertEqual(result["overall"]["mean"], 0.5)
        self.assertEqual(result["by_cohort"]["dataset_eval"]["mean"], 0.5)

    def test_clap_seed_diversity_requires_two_seeds_per_prompt(self) -> None:
        records = [
            {
                "clip_id": "prompt-a__seed-42",
                "prompt_id": "prompt-a",
                "cohort": "dataset_eval",
                "prompt": "Prompt A",
                "seed": 42,
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "at least two seeds"):
            score.summarize_clap_seed_diversity(
                records,
                {"prompt-a__seed-42": [1.0, 0.0]},
            )

    def test_seed_diversity_dry_run_rejects_single_seed_run(self) -> None:
        self.make_run()
        arguments = [
            "score.py",
            "--run-name",
            "test-run",
            "--metrics",
            "clap_seed_diversity",
            "--dry-run",
        ]

        with (
            mock.patch.object(sys, "argv", arguments),
            self.assertRaisesRegex(RuntimeError, "at least two generation seeds"),
        ):
            score.main()

    def test_seed_diversity_cli_writes_prompt_level_metric(self) -> None:
        run_dir = self.make_run()
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["seeds"].append(43)
        config_path.write_text(json.dumps(config), encoding="utf-8")

        audio_path = run_dir / "audio" / "dataset-1__seed-43.wav"
        audio_path.write_bytes(b"generated-43")
        manifest_path = run_dir / "manifest.jsonl"
        second_record = json.loads(manifest_path.read_text(encoding="utf-8"))
        second_record["clip_id"] = "dataset-1__seed-43"
        second_record["seed"] = 43
        second_record["audio_path"] = "audio/dataset-1__seed-43.wav"
        with manifest_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(second_record) + "\n")

        arguments = [
            "score.py",
            "--run-name",
            "test-run",
            "--metrics",
            "clap_seed_diversity",
            "--device",
            "cpu",
        ]
        embeddings = {
            "dataset-1__seed-42": [1.0, 0.0],
            "dataset-1__seed-43": [0.0, 1.0],
        }
        fake_torch = SimpleNamespace(__version__="test")

        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
            mock.patch.object(score, "select_device", return_value="cpu"),
            mock.patch.object(
                score,
                "resolve_clap_checkpoint",
                return_value=SCORER_PATH,
            ),
            mock.patch.object(
                score,
                "score_clap",
                return_value=({}, embeddings),
            ) as clap,
            redirect_stdout(io.StringIO()),
        ):
            score.main()

        self.assertFalse(clap.call_args.kwargs["include_text_scores"])
        output = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        diversity = output["metrics"]["clap_seed_diversity"]
        self.assertEqual(diversity["overall"]["mean"], 1.0)
        self.assertEqual(
            diversity["by_prompt"]["dataset-1"]["pair_count"],
            1,
        )
        clip_records = [
            json.loads(line)
            for line in (run_dir / "clip_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(clip_records), 2)
        self.assertTrue(
            all(record["metrics"] == {} for record in clip_records)
        )

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

    def test_validates_prepared_fad_reference_corpus(self) -> None:
        corpus_dir, records = self.make_fad_corpus()

        corpus = score.validate_fad_reference_corpus(corpus_dir)

        self.assertEqual(corpus["reference_set"], "human-fma-lofi-v1")
        self.assertEqual(corpus["reference_count"], len(records))
        self.assertEqual(
            corpus["manifest_sha256"],
            score.sha256_file(corpus_dir / "manifest.jsonl"),
        )
        self.assertEqual(len(corpus["audio_paths"]), len(records))

    def test_validates_multiple_fad_reference_corpora(self) -> None:
        human_dir, _ = self.make_fad_corpus("human-fma-lofi-v1")
        synthetic_dir, _ = self.make_fad_corpus("musicgen-large-v1")

        corpora = score.validate_fad_reference_corpora(
            [human_dir, synthetic_dir]
        )

        self.assertEqual(
            [corpus["reference_set"] for corpus in corpora],
            ["human-fma-lofi-v1", "musicgen-large-v1"],
        )

    def test_cli_accepts_multiple_fad_reference_corpora(self) -> None:
        arguments = [
            "score.py",
            "--run-name",
            "test-run",
            "--fad-reference-corpus",
            "references/first",
            "--fad-reference-corpus",
            "references/second",
        ]

        with mock.patch.object(sys, "argv", arguments):
            args = score.parse_args()

        self.assertEqual(
            args.fad_reference_corpus,
            [Path("references/first"), Path("references/second")],
        )

    def test_rejects_duplicate_fad_reference_set_names(self) -> None:
        first_dir, _ = self.make_fad_corpus(
            "human-fma-lofi-v1",
            directory_name="first",
        )
        second_dir, _ = self.make_fad_corpus(
            "human-fma-lofi-v1",
            directory_name="second",
        )

        with self.assertRaisesRegex(RuntimeError, "Duplicate FAD reference_set"):
            score.validate_fad_reference_corpora([first_dir, second_dir])

    def test_rejects_bad_fad_manifest_checksum(self) -> None:
        corpus_dir, _ = self.make_fad_corpus()
        with (corpus_dir / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write("{}\n")

        with self.assertRaisesRegex(RuntimeError, "manifest checksum mismatch"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_bad_fad_attribution_checksum(self) -> None:
        corpus_dir, _ = self.make_fad_corpus()
        with (corpus_dir / "ATTRIBUTION.md").open("a", encoding="utf-8") as file:
            file.write("changed\n")

        with self.assertRaisesRegex(RuntimeError, "attribution checksum mismatch"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_bad_fad_audio_checksum(self) -> None:
        corpus_dir, records = self.make_fad_corpus()
        records[0]["audio_sha256"] = "0" * 64
        self.write_fad_manifest(corpus_dir, records)

        with self.assertRaisesRegex(RuntimeError, "audio checksum mismatch"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_missing_fad_audio(self) -> None:
        corpus_dir, _ = self.make_fad_corpus()
        (corpus_dir / "audio" / "track-0.wav").unlink()

        with self.assertRaisesRegex(RuntimeError, "audio is missing"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_duplicate_fad_reference_ids(self) -> None:
        corpus_dir, records = self.make_fad_corpus()
        records[1]["reference_id"] = records[0]["reference_id"]
        self.write_fad_manifest(corpus_dir, records)

        with self.assertRaisesRegex(RuntimeError, "Duplicate or invalid"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_fad_reference_set_mismatch(self) -> None:
        corpus_dir, records = self.make_fad_corpus()
        records[0]["reference_set"] = "musicgen-large-v1"
        self.write_fad_manifest(corpus_dir, records)

        with self.assertRaisesRegex(RuntimeError, "reference_set mismatch"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_fad_audio_path_traversal(self) -> None:
        corpus_dir, records = self.make_fad_corpus(count=1)
        outside_audio = self.root / "outside.wav"
        outside_audio.write_bytes(b"outside")
        records[0]["audio_path"] = "../outside.wav"
        records[0]["audio_sha256"] = score.sha256_file(outside_audio)
        self.write_fad_manifest(corpus_dir, records)

        with self.assertRaisesRegex(RuntimeError, "contains traversal"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_absolute_fad_audio_path(self) -> None:
        corpus_dir, records = self.make_fad_corpus(count=1)
        audio_path = corpus_dir / "audio" / "track-0.wav"
        records[0]["audio_path"] = str(audio_path.resolve())
        self.write_fad_manifest(corpus_dir, records)

        with self.assertRaisesRegex(RuntimeError, "must be relative"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_rejects_fad_audio_symlink(self) -> None:
        corpus_dir, records = self.make_fad_corpus(count=1)
        link_path = corpus_dir / "audio" / "link.wav"
        link_path.symlink_to("track-0.wav")
        records[0]["audio_path"] = "audio/link.wav"
        self.write_fad_manifest(corpus_dir, records)

        with self.assertRaisesRegex(RuntimeError, "cannot use symlinks"):
            score.validate_fad_reference_corpus(corpus_dir)

    def test_dry_run_prints_fad_reference_corpus(self) -> None:
        self.make_run()
        corpus_dir, records = self.make_fad_corpus()
        arguments = [
            "score.py",
            "--run-name",
            "test-run",
            "--metrics",
            "fad",
            "--fad-reference-corpus",
            str(corpus_dir),
            "--dry-run",
        ]
        output = io.StringIO()

        with mock.patch.object(sys, "argv", arguments), redirect_stdout(output):
            score.main()

        dry_run = json.loads(output.getvalue())
        self.assertEqual(
            dry_run["fad_reference_corpora"],
            [
                {
                    "path": str(corpus_dir.resolve()),
                    "reference_set": "human-fma-lofi-v1",
                    "track_count": len(records),
                }
            ],
        )

    def test_dry_run_uses_both_default_fad_reference_corpora(self) -> None:
        self.make_run()
        human_dir, human_records = self.make_fad_corpus("human-fma-lofi-v1")
        synthetic_dir, synthetic_records = self.make_fad_corpus(
            "musicgen-large-v1"
        )
        arguments = [
            "score.py",
            "--run-name",
            "test-run",
            "--metrics",
            "fad",
            "--dry-run",
        ]
        output = io.StringIO()

        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(
                score,
                "DEFAULT_FAD_REFERENCE_CORPORA",
                (human_dir, synthetic_dir),
            ),
            redirect_stdout(output),
        ):
            score.main()

        dry_run = json.loads(output.getvalue())
        self.assertEqual(
            [
                (corpus["reference_set"], corpus["track_count"])
                for corpus in dry_run["fad_reference_corpora"]
            ],
            [
                ("human-fma-lofi-v1", len(human_records)),
                ("musicgen-large-v1", len(synthetic_records)),
            ],
        )

    def test_score_config_records_fad_corpus_provenance(self) -> None:
        run_dir = self.make_run()
        _, _, records = score.load_run("test-run")
        corpus_dir, _ = self.make_fad_corpus()
        corpus = score.validate_fad_reference_corpus(corpus_dir)
        clip_records = score.make_clip_records(run_dir, records, {}, {}, {})
        args = SimpleNamespace(
            run_name="test-run",
            reference_dir=self.root / "paired",
            clap_batch_size=4,
        )

        config = score.make_score_config(
            args,
            run_dir,
            ["fad"],
            {},
            clip_records,
            None,
            "cpu",
            [corpus],
        )

        provenance = config["fad"]["reference_corpora"][0]
        self.assertEqual(provenance["reference_set"], "human-fma-lofi-v1")
        self.assertEqual(provenance["reference_count"], 2)
        self.assertEqual(provenance["config_sha256"], corpus["config_sha256"])
        self.assertEqual(provenance["manifest_sha256"], corpus["manifest_sha256"])
        self.assertEqual(
            provenance["attribution_sha256"],
            corpus["attribution_sha256"],
        )
        self.assertEqual(
            provenance["combined_audio_sha256"],
            corpus["combined_audio_sha256"],
        )

    def test_score_config_records_clap_seed_diversity_method(self) -> None:
        run_dir = self.make_run()
        _, _, records = score.load_run("test-run")
        clip_records = score.make_clip_records(run_dir, records, {}, {}, {})
        args = SimpleNamespace(
            run_name="test-run",
            reference_dir=self.root / "paired",
            clap_batch_size=4,
        )

        config = score.make_score_config(
            args,
            run_dir,
            ["clap_seed_diversity"],
            {},
            clip_records,
            SCORER_PATH,
            "cpu",
        )

        method = config["clap"]["seed_diversity"]
        self.assertEqual(method["distance"], "cosine_distance")
        self.assertEqual(
            method["pairing"],
            "all_unordered_seed_pairs_within_each_prompt",
        )

    def test_legacy_fad_uses_paired_dataset_references(self) -> None:
        run_dir = self.make_run()
        _, _, records = score.load_run("test-run")
        reference_path = self.root / "legacy-reference.wav"
        reference_path.write_bytes(b"reference")
        references = {
            "source-1": {
                "audio_path": reference_path,
            }
        }
        generated_embeddings = SimpleNamespace(shape=(3, 128))
        reference_embeddings = SimpleNamespace(shape=(2, 128))

        with (
            mock.patch.object(
                score,
                "vggish_embeddings",
                side_effect=[generated_embeddings, reference_embeddings],
            ) as embeddings,
            mock.patch.object(score, "frechet_distance", return_value=4.5),
        ):
            result = score.score_fad(run_dir, records, references, "cpu")

        self.assertEqual(result["value"], 4.5)
        self.assertEqual(result["generated_clip_count"], 1)
        self.assertEqual(result["reference_clip_count"], 1)
        self.assertEqual(embeddings.call_count, 2)
        self.assertEqual(
            embeddings.call_args_list[1].args[0],
            [reference_path],
        )

    def test_reuses_generated_fad_embeddings_across_corpora(self) -> None:
        run_dir = self.make_run()
        _, _, records = score.load_run("test-run")
        first_reference = self.root / "first.wav"
        second_reference = self.root / "second.wav"
        first_reference.write_bytes(b"first")
        second_reference.write_bytes(b"second")
        corpora = [
            {
                "reference_set": "human-fma-lofi-v1",
                "audio_paths": [first_reference],
            },
            {
                "reference_set": "musicgen-large-v1",
                "audio_paths": [second_reference],
            },
        ]
        generated_embeddings = SimpleNamespace(shape=(3, 128))
        first_embeddings = SimpleNamespace(shape=(2, 128))
        second_embeddings = SimpleNamespace(shape=(4, 128))

        with (
            mock.patch.object(
                score,
                "vggish_embeddings",
                side_effect=[
                    generated_embeddings,
                    first_embeddings,
                    second_embeddings,
                ],
            ) as embeddings,
            mock.patch.object(
                score,
                "frechet_distance",
                side_effect=[1.25, 2.5],
            ) as distance,
        ):
            result = score.score_fad_reference_corpora(
                run_dir,
                records,
                corpora,
                "cpu",
            )

        self.assertEqual(embeddings.call_count, 3)
        self.assertIs(distance.call_args_list[0].args[1], generated_embeddings)
        self.assertIs(distance.call_args_list[1].args[1], generated_embeddings)
        self.assertEqual(
            result["by_reference_set"]["human-fma-lofi-v1"]["value"],
            1.25,
        )
        self.assertEqual(
            result["by_reference_set"]["musicgen-large-v1"][
                "reference_embedding_count"
            ],
            4,
        )


if __name__ == "__main__":
    unittest.main()
