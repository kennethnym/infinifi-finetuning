from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


PREPARE_PATH = Path(__file__).resolve().parents[1] / "prepare.py"
SPEC = importlib.util.spec_from_file_location("prepare", PREPARE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache_dir = self.root / "curation"
        (self.cache_dir / "audio").mkdir(parents=True)
        self.original_audiocraft_root = prepare.AUDIOCRAFT_ROOT
        self.original_dataset_root = prepare.DATASET_ROOT
        prepare.AUDIOCRAFT_ROOT = self.root / "audiocraft"
        prepare.DATASET_ROOT = prepare.AUDIOCRAFT_ROOT / "dataset" / "lofi"

    def tearDown(self) -> None:
        prepare.AUDIOCRAFT_ROOT = self.original_audiocraft_root
        prepare.DATASET_ROOT = self.original_dataset_root
        self.temporary_directory.cleanup()

    def find_id(self, split: str, start: int = 0) -> str:
        index = start
        while prepare.split_for_track(f"track-{index}") != split:
            index += 1
        return f"track-{index}"

    def make_candidates(
        self,
        specs: list[tuple[str, str, bytes]],
    ) -> list[dict[str, object]]:
        candidates = []
        for index, (track_id, caption, audio) in enumerate(specs):
            path = self.cache_dir / "audio" / f"{prepare.filename_for_track(track_id)}.mp3"
            path.write_bytes(audio)
            candidates.append(
                {
                    "schema_version": prepare.CANDIDATE_SCHEMA_VERSION,
                    "id": track_id,
                    "original_caption": caption,
                    "cached_audio_path": path.relative_to(self.cache_dir).as_posix(),
                    "audio_sha256": prepare.sha256_bytes(audio),
                    "split": prepare.split_for_track(track_id),
                    "source_position": index,
                    "order": index,
                }
            )
        return candidates

    def make_scores(
        self,
        candidates: list[dict[str, object]],
        values: dict[str, float] | None = None,
    ) -> list[dict[str, object]]:
        values = values or {}
        return [
            {
                "schema_version": prepare.SCORE_SCHEMA_VERSION,
                "id": candidate["id"],
                "original_caption": candidate["original_caption"],
                "effective_caption": candidate["original_caption"],
                "original_clap_score": values.get(candidate["id"], 0.5),
                "effective_clap_score": values.get(candidate["id"], 0.5),
                "audio_sha256": candidate["audio_sha256"],
                "split": candidate["split"],
                "status": "eligible",
            }
            for candidate in candidates
        ]

    @staticmethod
    def successful_scorer(
        records: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        return {
            record["id"]: {
                "status": "eligible",
                "original_clap_score": 0.4,
                "effective_clap_score": (
                    0.8
                    if record["effective_caption"] != record["original_caption"]
                    else 0.4
                ),
            }
            for record in records
        }

    def test_cli_defaults_and_validation(self) -> None:
        args = prepare.parse_args([])
        self.assertEqual(args.candidate_count, 20_000)
        self.assertEqual(args.train_count, 6_000)
        self.assertEqual(args.valid_count, 750)
        prepare.validate_args(args)

        args.clap_batch_size = 0
        with self.assertRaisesRegex(RuntimeError, "--clap-batch-size"):
            prepare.validate_args(args)

        args = prepare.parse_args(
            ["--candidate-count", "10", "--train-count", "8", "--valid-count", "3"]
        )
        with self.assertRaisesRegex(RuntimeError, "at least"):
            prepare.validate_args(args)

    def test_split_is_deterministic_sha256_partition(self) -> None:
        track_id = "stable/source/id"
        digest = prepare.hashlib.sha256(track_id.encode()).digest()
        expected = (
            "train"
            if int.from_bytes(digest[:8], "big") / 2**64 < prepare.TRAIN_SIZE
            else "eval"
        )
        self.assertEqual(prepare.split_for_track(track_id), expected)
        self.assertEqual(prepare.split_for_track(track_id), expected)

    def test_loads_non_empty_frozen_evaluation_ids(self) -> None:
        path = self.root / "prompts.jsonl"
        prepare.write_jsonl(
            path,
            [
                {"source_id": "frozen-1"},
                {"source_id": ""},
                {"source_id": None},
                {"source_id": " frozen-2 "},
            ],
        )
        self.assertEqual(
            prepare.load_frozen_evaluation_ids(path),
            {"frozen-1", "frozen-2"},
        )

    def test_candidate_count_is_after_filters_and_duplicate_ids(self) -> None:
        rows = [
            {"id": "ignored", "prompt": "funky beat", "audio": {"bytes": b"x"}},
            {"id": "a", "prompt": "quiet piano", "audio": {"bytes": b"a"}},
            {"id": "a", "prompt": "duplicate id", "audio": {"bytes": b"a2"}},
            {"id": "b", "prompt": "soft guitar", "audio": {"bytes": b"b"}},
        ]
        with redirect_stdout(io.StringIO()):
            candidates = prepare.collect_candidates(
                self.cache_dir,
                2,
                dataset_factory=lambda: iter(rows),
            )
        self.assertEqual([record["id"] for record in candidates], ["a", "b"])
        self.assertEqual(
            [record["source_position"] for record in candidates], [1, 3]
        )

    def test_candidate_collection_resumes_and_reuses_verified_audio(self) -> None:
        rows = [
            {"id": "a", "prompt": "quiet piano", "audio": {"bytes": b"a"}},
            {"id": "b", "prompt": "soft guitar", "audio": {"bytes": b"b"}},
            {"id": "c", "prompt": "vinyl rain", "audio": {"bytes": b"c"}},
        ]
        with redirect_stdout(io.StringIO()):
            first = prepare.collect_candidates(
                self.cache_dir, 2, dataset_factory=lambda: iter(rows)
            )
            resumed = prepare.collect_candidates(
                self.cache_dir, 3, dataset_factory=lambda: iter(rows)
            )
        self.assertEqual([record["id"] for record in first], ["a", "b"])
        self.assertEqual([record["id"] for record in resumed], ["a", "b", "c"])

        def should_not_load() -> object:
            raise AssertionError("completed candidate cache should avoid source loading")

        with redirect_stdout(io.StringIO()):
            reused = prepare.collect_candidates(
                self.cache_dir, 3, dataset_factory=should_not_load
            )
        self.assertEqual(reused, resumed)

    def test_recovers_audio_written_before_candidate_journal(self) -> None:
        track_id = "orphan"
        audio = b"orphan-audio"
        path = (
            self.cache_dir / "audio" / f"{prepare.filename_for_track(track_id)}.mp3"
        )
        path.write_bytes(audio)
        rows = [
            {
                "id": track_id,
                "prompt": "quiet rain",
                "audio": {"bytes": audio},
            }
        ]
        with redirect_stdout(io.StringIO()):
            candidates = prepare.collect_candidates(
                self.cache_dir, 1, dataset_factory=lambda: iter(rows)
            )
        self.assertEqual(candidates[0]["id"], track_id)

    def test_cache_config_mismatch_refuses_to_mix_results(self) -> None:
        first = prepare.build_cache_config(10)
        prepare.ensure_cache_config(self.cache_dir, first)
        second = prepare.build_cache_config(11)
        with self.assertRaisesRegex(RuntimeError, "Incompatible curation cache"):
            prepare.ensure_cache_config(self.cache_dir, second)

    def test_selection_only_config_change_preserves_candidate_cache(self) -> None:
        expected = prepare.build_cache_config(10)
        previous = json.loads(json.dumps(expected))
        previous["script"]["version"] = 2
        previous["selection_schema_version"] = 1
        previous.pop("deduplication")
        prepare.write_json(self.cache_dir / "config.json", previous)
        marker = self.cache_dir / "candidates.jsonl"
        marker.write_text("candidate cache marker\n", encoding="utf-8")

        with redirect_stdout(io.StringIO()):
            prepare.ensure_cache_config(self.cache_dir, expected)

        self.assertEqual(prepare.read_json(self.cache_dir / "config.json"), expected)
        self.assertEqual(
            marker.read_text(encoding="utf-8"), "candidate cache marker\n"
        )

    def test_override_validation_and_application(self) -> None:
        candidates = self.make_candidates(
            [("a", "caption a", b"a"), ("b", "caption b", b"b")]
        )
        path = self.cache_dir / "manual.jsonl"
        prepare.write_jsonl(
            path,
            [
                {"id": "a", "action": "drop", "reason": "mismatch"},
                {
                    "id": "b",
                    "action": "rewrite",
                    "caption": "better caption",
                    "reason": "audited",
                },
            ],
        )
        overrides = prepare.load_overrides(path, candidates)
        scores = self.make_scores(candidates)
        scores[1]["effective_caption"] = "better caption"
        scores[1]["effective_clap_score"] = 0.9
        selection = prepare.build_selection(
            candidates, scores, overrides, set(), 1, 1
        )
        by_id = {record["id"]: record for record in selection}
        self.assertEqual(by_id["a"]["decision"], "dropped_by_override")
        self.assertEqual(by_id["a"]["rejection_reason"], "mismatch")
        self.assertEqual(by_id["b"]["effective_caption"], "better caption")

        prepare.write_jsonl(
            path,
            [
                {"id": "a", "action": "drop"},
                {"id": "a", "action": "drop"},
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "Duplicate override"):
            prepare.load_overrides(path, candidates)

        prepare.write_jsonl(path, [{"id": "b", "action": "rewrite", "caption": " "}])
        with self.assertRaisesRegex(RuntimeError, "non-empty caption"):
            prepare.load_overrides(path, candidates)

    def test_rewrite_change_rescores_and_preserves_both_scores(self) -> None:
        candidates = self.make_candidates([("a", "old caption", b"a")])
        calls: list[list[dict[str, object]]] = []

        def scorer(
            records: list[dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            calls.append(records)
            return self.successful_scorer(records)

        with redirect_stdout(io.StringIO()):
            initial = prepare.score_candidates(
                candidates,
                self.cache_dir,
                {},
                1,
                None,
                "cpu",
                score_batch=scorer,
            )
            rewritten = prepare.score_candidates(
                candidates,
                self.cache_dir,
                {"a": {"action": "rewrite", "caption": "new caption"}},
                1,
                None,
                "cpu",
                score_batch=scorer,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0]["original_caption"], "old caption")
        self.assertEqual(calls[1][0]["effective_caption"], "new caption")
        self.assertEqual(initial[0]["original_clap_score"], 0.4)
        self.assertEqual(rewritten[0]["original_clap_score"], 0.4)
        self.assertEqual(rewritten[0]["effective_clap_score"], 0.8)

    def test_scoring_resumes_completed_batches(self) -> None:
        candidates = self.make_candidates(
            [("a", "caption a", b"a"), ("b", "caption b", b"b")]
        )
        first_calls: list[str] = []

        def interrupted(
            records: list[dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            first_calls.append(records[0]["id"])
            if records[0]["id"] == "b":
                raise RuntimeError("simulated failure")
            return self.successful_scorer(records)

        with (
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "simulated failure"),
        ):
            prepare.score_candidates(
                candidates,
                self.cache_dir,
                {},
                1,
                None,
                "cpu",
                score_batch=interrupted,
            )

        resumed_calls: list[str] = []

        def resumed(
            records: list[dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            resumed_calls.extend(record["id"] for record in records)
            return self.successful_scorer(records)

        with redirect_stdout(io.StringIO()):
            scores = prepare.score_candidates(
                candidates,
                self.cache_dir,
                {},
                1,
                None,
                "cpu",
                score_batch=resumed,
            )
        self.assertEqual(first_calls, ["a", "b"])
        self.assertEqual(resumed_calls, ["b"])
        self.assertEqual([record["id"] for record in scores], ["a", "b"])

        with redirect_stdout(io.StringIO()):
            prepare.score_candidates(
                candidates,
                self.cache_dir,
                {},
                1,
                None,
                "cpu",
                score_batch=lambda _records: self.fail("should reuse scores"),
            )

    def test_eval_exclusion_audio_and_caption_deduplication(self) -> None:
        candidates = self.make_candidates(
            [
                ("audio-low", "unique low", b"same"),
                ("audio-high", "unique high", b"same"),
                ("caption-z", "  SAME   Caption ", b"z"),
                ("caption-a", "same caption", b"a"),
                ("frozen", "frozen caption", b"f"),
            ]
        )
        scores = self.make_scores(
            candidates,
            {
                "audio-low": 0.1,
                "audio-high": 0.9,
                "caption-z": 0.8,
                "caption-a": 0.8,
                "frozen": 1.0,
            },
        )
        selection = prepare.build_selection(
            candidates, scores, {}, {"frozen"}, 1, 1
        )
        by_id = {record["id"]: record for record in selection}
        self.assertEqual(by_id["audio-low"]["decision"], "duplicate_audio_sha256")
        self.assertEqual(by_id["audio-low"]["retained_id"], "audio-high")
        self.assertEqual(
            by_id["caption-z"]["decision"], "duplicate_normalized_caption"
        )
        self.assertEqual(by_id["caption-z"]["retained_id"], "caption-a")
        self.assertEqual(
            by_id["frozen"]["decision"], "excluded_frozen_evaluation"
        )

    def test_ranking_is_independent_and_score_ties_use_id(self) -> None:
        train_a = self.find_id("train", 0)
        train_b = self.find_id("train", int(train_a.split("-")[1]) + 1)
        eval_a = self.find_id("eval", 0)
        eval_b = self.find_id("eval", int(eval_a.split("-")[1]) + 1)
        candidates = self.make_candidates(
            [
                (train_b, "train b", b"tb"),
                (train_a, "train a", b"ta"),
                (eval_b, "eval b", b"eb"),
                (eval_a, "eval a", b"ea"),
            ]
        )
        scores = self.make_scores(
            candidates,
            {track_id: 0.5 for track_id in (train_a, train_b, eval_a, eval_b)},
        )
        selection = prepare.build_selection(
            candidates, scores, {}, set(), 1, 1
        )
        selected = {
            record["split"]: record["id"]
            for record in selection
            if record["selected"]
        }
        self.assertEqual(selected["train"], min(train_a, train_b))
        self.assertEqual(selected["eval"], min(eval_a, eval_b))

    def test_near_identical_captions_keep_highest_scoring_candidate(self) -> None:
        candidates = self.make_candidates(
            [
                (
                    "caption-low",
                    "mellow piano lofi beat with soft vinyl crackle",
                    b"low",
                ),
                (
                    "caption-high",
                    "mellow piano lofi beat with soft vinyl crackles",
                    b"high",
                ),
                (
                    "different",
                    "mellow guitar lofi beat with hard drums",
                    b"different",
                ),
            ]
        )
        scores = self.make_scores(
            candidates,
            {"caption-low": 0.5, "caption-high": 0.9, "different": 0.8},
        )

        selection = prepare.build_selection(
            candidates, scores, {}, set(), 1, 1
        )
        by_id = {record["id"]: record for record in selection}

        self.assertEqual(
            by_id["caption-low"]["decision"], "near_duplicate_caption"
        )
        self.assertEqual(by_id["caption-low"]["retained_id"], "caption-high")
        self.assertGreaterEqual(
            by_id["caption-low"]["caption_similarity"],
            prepare.NEAR_CAPTION_SIMILARITY_THRESHOLD,
        )
        self.assertNotEqual(
            by_id["different"]["decision"], "near_duplicate_caption"
        )

    def test_insufficient_eligible_candidates_has_useful_error(self) -> None:
        train_id = self.find_id("train")
        eval_id = self.find_id("eval")
        candidates = self.make_candidates(
            [(train_id, "train", b"t"), (eval_id, "eval", b"e")]
        )
        selection = prepare.build_selection(
            candidates,
            self.make_scores(candidates),
            {train_id: {"action": "drop"}},
            set(),
            1,
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "Insufficient eligible"):
            prepare.require_exact_selection(selection, 1, 1)

    @staticmethod
    def keyword_extractor(
        prompts: list[str],
    ) -> list[list[tuple[str, float]]]:
        return [[("lofi", 1.0)] for _prompt in prompts]

    @staticmethod
    def manifest_generator(
        dataset_root: Path,
        split: str,
        manifest_path: Path,
        expected: int,
    ) -> None:
        paths = sorted((dataset_root / split).glob("*.mp3"))
        prepare.write_jsonl(
            manifest_path,
            [{"path": str(path)} for path in paths],
        )
        if len(paths) != expected:
            raise RuntimeError("unexpected test fixture count")

    def test_materializes_exact_counts_and_effective_caption(self) -> None:
        train_id = self.find_id("train")
        eval_id = self.find_id("eval")
        candidates = self.make_candidates(
            [(train_id, "train old", b"train"), (eval_id, "eval", b"eval")]
        )
        scores = self.make_scores(candidates)
        scores[0]["effective_caption"] = "train rewritten"
        scores[0]["effective_clap_score"] = 0.9
        selection = prepare.build_selection(
            candidates,
            scores,
            {train_id: {"action": "rewrite", "caption": "train rewritten"}},
            set(),
            1,
            1,
        )
        prepare.materialize_selection(
            selection,
            candidates,
            self.cache_dir,
            1,
            1,
            keyword_extractor=self.keyword_extractor,
            manifest_generator=self.manifest_generator,
        )
        self.assertEqual(
            len(list((prepare.DATASET_ROOT / "train").glob("*.mp3"))), 1
        )
        self.assertEqual(
            len(list((prepare.DATASET_ROOT / "eval").glob("*.mp3"))), 1
        )
        train_metadata = json.loads(
            next((prepare.DATASET_ROOT / "train").glob("*.json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(train_metadata["description"], "train rewritten")
        self.assertEqual(train_metadata["name"], train_id)
        self.assertEqual(
            len(
                prepare.read_jsonl(
                    prepare.AUDIOCRAFT_ROOT / "egs" / "train" / "data.jsonl"
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                prepare.read_jsonl(
                    prepare.AUDIOCRAFT_ROOT / "egs" / "eval" / "data.jsonl"
                )
            ),
            1,
        )

    def test_failed_staging_leaves_existing_final_dataset_intact(self) -> None:
        train_id = self.find_id("train")
        eval_id = self.find_id("eval")
        candidates = self.make_candidates(
            [(train_id, "train", b"train"), (eval_id, "eval", b"eval")]
        )
        selection = prepare.build_selection(
            candidates, self.make_scores(candidates), {}, set(), 1, 1
        )
        prepare.DATASET_ROOT.mkdir(parents=True)
        old_file = prepare.DATASET_ROOT / "old.txt"
        old_file.write_text("keep me", encoding="utf-8")

        def fail_manifest(
            _dataset_root: Path,
            _split: str,
            _manifest_path: Path,
            _expected: int,
        ) -> None:
            raise RuntimeError("manifest failed")

        with self.assertRaisesRegex(RuntimeError, "manifest failed"):
            prepare.materialize_selection(
                selection,
                candidates,
                self.cache_dir,
                1,
                1,
                keyword_extractor=self.keyword_extractor,
                manifest_generator=fail_manifest,
            )
        self.assertEqual(old_file.read_text(encoding="utf-8"), "keep me")

    def test_scoring_failure_leaves_existing_final_dataset_intact(self) -> None:
        candidates = self.make_candidates([("a", "caption", b"audio")])
        prepare.DATASET_ROOT.mkdir(parents=True)
        old_file = prepare.DATASET_ROOT / "old.txt"
        old_file.write_text("keep me", encoding="utf-8")

        def fail_scoring(
            _records: list[dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            raise RuntimeError("scoring failed")

        with (
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "scoring failed"),
        ):
            prepare.score_candidates(
                candidates,
                self.cache_dir,
                {},
                1,
                None,
                "cpu",
                score_batch=fail_scoring,
            )
        self.assertEqual(old_file.read_text(encoding="utf-8"), "keep me")

    def test_summary_records_selection_digest_and_counts(self) -> None:
        train_id = self.find_id("train")
        eval_id = self.find_id("eval")
        candidates = self.make_candidates(
            [(train_id, "train", b"train"), (eval_id, "eval", b"eval")]
        )
        selection = prepare.build_selection(
            candidates, self.make_scores(candidates), {}, set(), 1, 1
        )
        config = prepare.build_cache_config(2)
        summary = prepare.write_selection_artifacts(
            self.cache_dir, candidates, selection, config, 1, 1
        )
        self.assertEqual(summary["actual_counts"]["train"], 1)
        self.assertEqual(summary["actual_counts"]["valid"], 1)
        self.assertEqual(
            summary["selection_sha256"],
            prepare.sha256_file(self.cache_dir / "selection.jsonl"),
        )

    def test_run_logs_every_pipeline_stage(self) -> None:
        train_id = self.find_id("train")
        eval_id = self.find_id("eval")
        candidates = self.make_candidates(
            [(train_id, "train", b"train"), (eval_id, "eval", b"eval")]
        )
        scores = self.make_scores(candidates)
        args = SimpleNamespace(
            candidate_count=2,
            train_count=1,
            valid_count=1,
            clap_batch_size=1,
            clap_checkpoint=None,
            cache_dir=self.cache_dir,
            overrides=None,
            device="cpu",
        )
        output = io.StringIO()
        with (
            mock.patch.object(prepare, "check_audiocraft_checkout"),
            mock.patch.object(prepare, "ensure_cache_config"),
            mock.patch.object(prepare, "collect_candidates", return_value=candidates),
            mock.patch.object(prepare, "score_candidates", return_value=scores),
            mock.patch.object(
                prepare, "load_frozen_evaluation_ids", return_value=set()
            ),
            mock.patch.object(
                prepare, "write_selection_artifacts", return_value={}
            ),
            mock.patch.object(prepare, "materialize_selection"),
            redirect_stdout(output),
        ):
            prepare.run(args)

        logged = output.getvalue()
        for stage in range(1, 9):
            self.assertIn(f"stage {stage}/8", logged)
        self.assertIn("candidate pool ready", logged)
        self.assertIn("selection ready", logged)


if __name__ == "__main__":
    unittest.main()
