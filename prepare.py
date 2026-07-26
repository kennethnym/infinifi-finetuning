import argparse
from contextlib import nullcontext
from difflib import SequenceMatcher
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable
import uuid


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SAMPLE_RATE = 44_100
TRAIN_SIZE = 0.8
SHUFFLE_SEED = 42
BATCH_SIZE = 32
REPO = "vikhyatk/lofi"
DATASET_REVISION = "966a2d3065aac26c0385b4ef2d50983c0429a305"

CLAP_REPOSITORY = "lukewys/laion_clap"
CLAP_CHECKPOINT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CHECKPOINT_SHA256 = (
    "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
)
CLAP_SCORING_SEED = 20260715
CLAP_SAMPLE_RATE = 48_000
CLAP_DURATION_SECONDS = 30
MIN_AUDIO_DURATION_SECONDS = 1

CURATION_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
SCORE_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 2
SPLIT_ALGORITHM = "sha256-first-8-u64-v1"
NEAR_CAPTION_DEDUPLICATION_ALGORITHM = "ordered-token-one-edit-sequence-v1"
NEAR_CAPTION_SIMILARITY_THRESHOLD = 0.90
NEAR_CAPTION_MIN_CHARACTERS = 8
NEAR_CAPTION_LEXICAL_WINDOW = 8

DEFAULT_CANDIDATE_COUNT = 20_000
DEFAULT_TRAIN_COUNT = 6_000
DEFAULT_VALID_COUNT = 750
DEFAULT_CLAP_BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIOCRAFT_ROOT = PROJECT_ROOT / "audiocraft"
DATASET_ROOT = AUDIOCRAFT_ROOT / "dataset" / "lofi"
PROMPTS_PATH = PROJECT_ROOT / "eval" / "prompts.jsonl"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "curation"
SPLITS = ("train", "eval")

IGNORE_WORDS = {"cello", "funky"}
INSTRUMENTS = ("piano", "guitar", "violin", "flute", "xylophone")
MOODS = (
    "nostalgic",
    "chill",
    "chilling",
    "uplift",
    "dreamy",
    "exhausted",
    "intimate",
    "dramatic",
    "frustrated",
    "uplifting",
    "soulful",
    "calm",
    "zen",
    "lively",
    "cozy",
    "peaceful",
    "sensual",
    "playful",
    "joyous",
    "passionate",
    "enigmatic",
    "soothing",
)


def log_progress(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate vikhyatk/lofi with CLAP and materialize exact AudioCraft "
            "training and validation sets."
        )
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
        help="Eligible unique candidates to cache (default: 20000).",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=DEFAULT_TRAIN_COUNT,
        help="Exact selected training tracks (default: 6000).",
    )
    parser.add_argument(
        "--valid-count",
        type=int,
        default=DEFAULT_VALID_COUNT,
        help="Exact selected validation tracks (default: 750).",
    )
    parser.add_argument(
        "--clap-batch-size",
        type=int,
        default=DEFAULT_CLAP_BATCH_SIZE,
        help="CLAP inference batch size (default: 32).",
    )
    parser.add_argument(
        "--clap-checkpoint",
        type=Path,
        help=(
            "Local pinned LAION-CLAP checkpoint. The pinned checkpoint is "
            "downloaded through Hugging Face when omitted."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Persistent curation cache (default: <project>/curation).",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        help=(
            "Optional JSONL drop/rewrite file. Defaults to "
            "<cache-dir>/overrides.jsonl when that file exists."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for CLAP, or 'auto' to prefer CUDA.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for option, value in (
        ("--candidate-count", args.candidate_count),
        ("--train-count", args.train_count),
        ("--valid-count", args.valid_count),
        ("--clap-batch-size", args.clap_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"{option} must be greater than zero.")
    if args.train_count + args.valid_count > args.candidate_count:
        raise RuntimeError(
            "--candidate-count must be at least --train-count + --valid-count."
        )
    if not isinstance(args.device, str) or not args.device.strip():
        raise RuntimeError("--device must be a non-empty torch device or 'auto'.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raise_disk_error(error: OSError, path: Path) -> None:
    if error.errno == errno.ENOSPC:
        raise RuntimeError(f"Insufficient disk space while writing {path}.") from error
    raise error


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        _raise_disk_error(error, path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        _raise_disk_error(error, path)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            file.flush()
            os.fsync(file.fileno())
    except OSError as error:
        _raise_disk_error(error, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Required JSON file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return value


def read_jsonl(
    path: Path,
    *,
    missing_ok: bool = False,
    repair_truncated_tail: bool = False,
) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if missing_ok:
            return []
        raise RuntimeError(f"Required JSONL file not found: {path}.")

    lines = text.splitlines()
    records: list[dict[str, Any]] = []
    repaired = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            is_truncated_tail = (
                repair_truncated_tail
                and index == len(lines) - 1
                and not text.endswith("\n")
            )
            if is_truncated_tail:
                repaired = True
                break
            raise RuntimeError(
                f"Invalid JSON in {path} at line {index + 1}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Expected an object in {path} at line {index + 1}."
            )
        records.append(record)
    if repaired:
        log_progress(f"discarding interrupted trailing record from {path}")
        write_jsonl(path, records)
    return records


def build_cache_config(candidate_count: int) -> dict[str, Any]:
    return {
        "schema_version": CURATION_SCHEMA_VERSION,
        "script": {"name": "prepare.py", "version": 3},
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "dataset": {
            "repository": REPO,
            "revision": DATASET_REVISION,
            "split": "train",
            "streaming": True,
            "shuffle_seed": SHUFFLE_SEED,
            "source_sample_rate": SAMPLE_RATE,
        },
        "candidate_count": candidate_count,
        "ignored_words": sorted(IGNORE_WORDS),
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "train_fraction": TRAIN_SIZE,
        },
        "deduplication": {
            "audio": "sha256",
            "normalized_caption": "trim-lowercase-collapse-whitespace",
            "near_caption": {
                "algorithm": NEAR_CAPTION_DEDUPLICATION_ALGORITHM,
                "similarity_threshold": NEAR_CAPTION_SIMILARITY_THRESHOLD,
                "minimum_characters": NEAR_CAPTION_MIN_CHARACTERS,
                "lexical_window": NEAR_CAPTION_LEXICAL_WINDOW,
            },
        },
        "clap": {
            "repository": CLAP_REPOSITORY,
            "checkpoint": CLAP_CHECKPOINT_NAME,
            "checkpoint_sha256": CLAP_CHECKPOINT_SHA256,
            "audio_model": "HTSAT-base",
            "fusion": False,
            "sample_rate": CLAP_SAMPLE_RATE,
            "duration_seconds": CLAP_DURATION_SECONDS,
            "minimum_audio_duration_seconds": MIN_AUDIO_DURATION_SECONDS,
            "scoring_seed": CLAP_SCORING_SEED,
            "similarity": "cosine",
        },
    }


def ensure_cache_config(cache_dir: Path, expected: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "audio").mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "config.json"
    if not config_path.exists():
        write_json(config_path, expected)
        return
    actual = read_json(config_path)
    if actual == expected:
        return
    identity_and_scoring_fields = (
        "schema_version",
        "candidate_schema_version",
        "score_schema_version",
        "dataset",
        "candidate_count",
        "ignored_words",
        "split",
        "clap",
    )
    incompatible = [
        field
        for field in identity_and_scoring_fields
        if actual.get(field) != expected.get(field)
    ]
    if incompatible:
        raise RuntimeError(
            f"Incompatible curation cache config in {config_path}; differing "
            f"fields: {', '.join(incompatible)}. Use another --cache-dir or "
            "deliberately remove the old cache."
        )
    log_progress(
        "updating compatible selection-only cache metadata without discarding "
        "candidate or CLAP caches"
    )
    write_json(config_path, expected)


def check_audiocraft_checkout() -> None:
    audio_dataset_module = AUDIOCRAFT_ROOT / "audiocraft" / "data" / "audio_dataset.py"
    if not audio_dataset_module.is_file():
        raise RuntimeError(
            "The AudioCraft submodule is not initialized. Run "
            "`git submodule update --init --recursive` first."
        )


def dataset_config_text() -> str:
    return f"""# @package __global__

datasource:
    max_sample_rate: {SAMPLE_RATE}
    max_channels: 1

    train: egs/train
    valid: egs/eval
    evaluate: egs/eval
    generate: egs/train
"""


def write_dataset_config() -> None:
    config_path = AUDIOCRAFT_ROOT / "config" / "dset" / "audio" / "lofi.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    temporary_path.write_text(dataset_config_text(), encoding="utf-8")
    temporary_path.replace(config_path)


def split_for_track(track_id: str) -> str:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    split_value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if split_value < TRAIN_SIZE else "eval"


def filename_for_track(track_id: str) -> str:
    readable_id = re.sub(r"[^A-Za-z0-9._-]+", "-", track_id).strip(".-")
    readable_id = readable_id[:80] or "track"
    digest = hashlib.sha256(track_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable_id}-{digest}"


def normalize_caption(caption: str) -> str:
    return " ".join(caption.strip().lower().split())


def lexical_caption(caption: str) -> str:
    return " ".join(re.findall(r"\w+", caption.lower(), flags=re.UNICODE))


def near_caption_similarity(left: str, right: str) -> float:
    left_lexical = lexical_caption(left)
    right_lexical = lexical_caption(right)
    if (
        len(left_lexical) < NEAR_CAPTION_MIN_CHARACTERS
        or len(right_lexical) < NEAR_CAPTION_MIN_CHARACTERS
    ):
        return 0.0
    return SequenceMatcher(
        None,
        left_lexical,
        right_lexical,
        autojunk=False,
    ).ratio()


def _caption_token_signatures(caption: str) -> set[tuple[str, ...]]:
    tokens = tuple(lexical_caption(caption).split())
    signatures = {tokens}
    if len(tokens) >= 3:
        signatures.update(tokens[:index] + tokens[index + 1 :] for index in range(len(tokens)))
    return signatures


def find_near_duplicate_caption_groups(
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if len(records) < 2:
        return []

    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    lexical = [lexical_caption(record["effective_caption"]) for record in records]
    candidate_pairs: set[tuple[int, int]] = set()
    by_signature: dict[tuple[str, ...], list[int]] = {}
    for index, record in enumerate(records):
        for signature in _caption_token_signatures(record["effective_caption"]):
            if signature:
                by_signature.setdefault(signature, []).append(index)
    for indices in by_signature.values():
        ordered_indices = sorted(
            set(indices),
            key=lambda index: (lexical[index], records[index]["id"]),
        )
        for offset, left in enumerate(ordered_indices):
            for right in ordered_indices[
                offset + 1 : offset + 1 + NEAR_CAPTION_LEXICAL_WINDOW
            ]:
                candidate_pairs.add((min(left, right), max(left, right)))

    lexical_order = sorted(
        range(len(records)),
        key=lambda index: (lexical[index], records[index]["id"]),
    )
    for offset, left in enumerate(lexical_order):
        for right in lexical_order[
            offset + 1 : offset + 1 + NEAR_CAPTION_LEXICAL_WINDOW
        ]:
            candidate_pairs.add((min(left, right), max(left, right)))

    for left, right in sorted(candidate_pairs):
        left_length = len(lexical[left])
        right_length = len(lexical[right])
        if min(left_length, right_length) < NEAR_CAPTION_MIN_CHARACTERS:
            continue
        maximum_possible = (
            2 * min(left_length, right_length) / (left_length + right_length)
        )
        if maximum_possible < NEAR_CAPTION_SIMILARITY_THRESHOLD:
            continue
        similarity = SequenceMatcher(
            None,
            lexical[left],
            lexical[right],
            autojunk=False,
        ).ratio()
        if similarity >= NEAR_CAPTION_SIMILARITY_THRESHOLD:
            union(left, right)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return [group for group in grouped.values() if len(group) > 1]


def _candidate_audio_path(cache_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError("Candidate contains an invalid cached_audio_path.")
    supplied = Path(relative_path)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise RuntimeError(f"Candidate audio path must stay in its cache: {relative_path}")
    path = cache_dir / supplied
    if path.is_symlink():
        raise RuntimeError(f"Candidate audio cannot be a symlink: {path}")
    try:
        path.resolve().relative_to(cache_dir.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Candidate audio path escapes its cache: {relative_path}"
        ) from error
    return path


def validate_candidates(
    records: list[dict[str, Any]],
    cache_dir: Path,
    candidate_count: int,
) -> list[dict[str, Any]]:
    if len(records) > candidate_count:
        raise RuntimeError(
            f"Candidate cache has {len(records)} records but config requests "
            f"{candidate_count}."
        )
    seen_ids: set[str] = set()
    seen_positions: set[int] = set()
    previous_position = -1
    required = {
        "schema_version",
        "id",
        "original_caption",
        "cached_audio_path",
        "audio_sha256",
        "split",
        "source_position",
        "order",
    }
    for order, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise RuntimeError(
                f"Candidate cache record {order + 1} is missing {sorted(missing)}."
            )
        track_id = record["id"]
        caption = record["original_caption"]
        position = record["source_position"]
        if record["schema_version"] != CANDIDATE_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported candidate schema for {track_id!r}.")
        if not isinstance(track_id, str) or not track_id or track_id in seen_ids:
            raise RuntimeError(f"Duplicate or invalid candidate id: {track_id!r}.")
        if not isinstance(caption, str) or not caption.strip():
            raise RuntimeError(f"Candidate {track_id} has an empty caption.")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
            or position in seen_positions
            or position <= previous_position
        ):
            raise RuntimeError(f"Candidate {track_id} has an invalid source position.")
        if record["order"] != order:
            raise RuntimeError(f"Candidate {track_id} has an invalid deterministic order.")
        if record["split"] != split_for_track(track_id):
            raise RuntimeError(f"Candidate {track_id} has an invalid split.")
        audio_sha256 = record["audio_sha256"]
        if (
            not isinstance(audio_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", audio_sha256) is None
        ):
            raise RuntimeError(f"Candidate {track_id} has an invalid audio SHA-256.")
        audio_path = _candidate_audio_path(
            cache_dir, record["cached_audio_path"]
        )
        if not audio_path.is_file():
            raise RuntimeError(f"Cached candidate audio is missing: {audio_path}")
        actual_sha256 = sha256_file(audio_path)
        if actual_sha256 != audio_sha256:
            raise RuntimeError(
                f"Cached audio checksum mismatch for {track_id}: expected "
                f"{audio_sha256}, got {actual_sha256}."
            )
        seen_ids.add(track_id)
        seen_positions.add(position)
        previous_position = position
    return records


def load_source_dataset() -> Iterable[dict[str, Any]]:
    try:
        import datasets
    except ImportError as error:
        raise RuntimeError(
            "Dataset preparation dependencies are missing; install "
            "pipeline-requirements.txt."
        ) from error
    dataset = datasets.load_dataset(
        REPO,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=SHUFFLE_SEED)
    return dataset.cast_column(
        "audio",
        datasets.Audio(sampling_rate=SAMPLE_RATE, decode=False),
    )


def audio_bytes_from_row(row: dict[str, Any]) -> bytes:
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise RuntimeError("Source row contains invalid audio metadata.")
    value = audio.get("bytes")
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    path_value = audio.get("path")
    if isinstance(path_value, str) and path_value:
        try:
            return Path(path_value).read_bytes()
        except OSError as error:
            raise RuntimeError(f"Unable to read source audio {path_value}: {error}") from error
    raise RuntimeError("Source row contains neither audio bytes nor a readable path.")


def collect_candidates(
    cache_dir: Path,
    candidate_count: int,
    *,
    dataset_factory: Callable[[], Iterable[dict[str, Any]]] = load_source_dataset,
) -> list[dict[str, Any]]:
    candidates_path = cache_dir / "candidates.jsonl"
    records = read_jsonl(
        candidates_path,
        missing_ok=True,
        repair_truncated_tail=True,
    )
    records = validate_candidates(records, cache_dir, candidate_count)
    if len(records) == candidate_count:
        log_progress(f"reusing {candidate_count} verified cached candidates")
        return records

    seen_ids = {record["id"] for record in records}
    last_position = records[-1]["source_position"] if records else -1
    source = iter(dataset_factory())
    journal_buffer: list[dict[str, Any]] = []
    log_progress(
        f"collecting candidates {len(records) + 1}-{candidate_count} "
        f"from {REPO}@{DATASET_REVISION}"
    )
    for source_position, row in enumerate(source):
        if source_position <= last_position:
            continue
        track_id_value = row.get("id")
        prompt_value = row.get("prompt")
        if not isinstance(track_id_value, (str, int)):
            continue
        track_id = str(track_id_value).strip()
        if not track_id or track_id in seen_ids:
            continue
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            continue
        prompt = prompt_value.strip()
        if any(word in prompt.lower() for word in IGNORE_WORDS):
            continue

        audio_bytes = audio_bytes_from_row(row)
        relative_audio_path = Path("audio") / f"{filename_for_track(track_id)}.mp3"
        audio_path = cache_dir / relative_audio_path
        audio_sha256 = sha256_bytes(audio_bytes)
        if audio_path.exists():
            if not audio_path.is_file() or sha256_file(audio_path) != audio_sha256:
                raise RuntimeError(
                    f"Unexpected conflicting untracked candidate audio: {audio_path}"
                )
            log_progress(f"recovering interrupted candidate {track_id}")
        else:
            temporary_audio_path = audio_path.with_name(f".{audio_path.name}.tmp")
            try:
                temporary_audio_path.write_bytes(audio_bytes)
                temporary_audio_path.replace(audio_path)
            except OSError as error:
                temporary_audio_path.unlink(missing_ok=True)
                _raise_disk_error(error, audio_path)

        record = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "id": track_id,
            "original_caption": prompt,
            "cached_audio_path": relative_audio_path.as_posix(),
            "audio_sha256": audio_sha256,
            "split": split_for_track(track_id),
            "source_position": source_position,
            "order": len(records),
        }
        records.append(record)
        journal_buffer.append(record)
        seen_ids.add(track_id)
        if len(records) % 100 == 0 or len(records) == candidate_count:
            append_jsonl(candidates_path, journal_buffer)
            journal_buffer.clear()
            log_progress(f"collected {len(records)}/{candidate_count} candidates")
        if len(records) == candidate_count:
            write_jsonl(candidates_path, records)
            return records

    if journal_buffer:
        append_jsonl(candidates_path, journal_buffer)
    raise RuntimeError(
        f"The source stream ended after {len(records)} eligible unique candidates; "
        f"--candidate-count requested {candidate_count}."
    )


def resolve_overrides_path(supplied: Path | None, cache_dir: Path) -> Path | None:
    if supplied is not None:
        path = supplied.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Override file not found: {path}")
        return path
    default = cache_dir / "overrides.jsonl"
    return default if default.is_file() else None


def load_overrides(
    path: Path | None,
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    candidate_ids = {record["id"] for record in candidates}
    overrides: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        track_id = record.get("id")
        action = record.get("action")
        if not isinstance(track_id, str) or not track_id:
            raise RuntimeError(f"Override line {line_number} has an invalid id.")
        if track_id in overrides:
            raise RuntimeError(f"Duplicate override id: {track_id}.")
        if track_id not in candidate_ids:
            raise RuntimeError(f"Override id is not in the candidate pool: {track_id}.")
        if action not in {"drop", "rewrite"}:
            raise RuntimeError(
                f"Override {track_id} action must be 'drop' or 'rewrite'."
            )
        reason = record.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise RuntimeError(f"Override {track_id} has an invalid reason.")
        normalized: dict[str, Any] = {"action": action}
        if reason is not None:
            normalized["reason"] = reason.strip()
        if action == "rewrite":
            caption = record.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise RuntimeError(
                    f"Rewrite override {track_id} requires a non-empty caption."
                )
            normalized["caption"] = caption.strip()
        elif "caption" in record:
            raise RuntimeError(f"Drop override {track_id} cannot include a caption.")
        overrides[track_id] = normalized
    return overrides


def effective_caption(
    candidate: dict[str, Any],
    override: dict[str, Any] | None,
) -> str:
    if override is not None and override["action"] == "rewrite":
        return override["caption"]
    return candidate["original_caption"]


def select_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    try:
        torch.empty(0, device=requested)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(f"Invalid or unavailable torch device: {requested}") from error
    return requested


def seed_everything(torch: Any, seed: int) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("CLAP scoring requires numpy.") from error
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_clap_checkpoint(supplied: Path | None) -> Path:
    if supplied is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required to download CLAP.") from error
        checkpoint = Path(
            hf_hub_download(
                repo_id=CLAP_REPOSITORY,
                filename=CLAP_CHECKPOINT_NAME,
            )
        )
    else:
        checkpoint = supplied.expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"CLAP checkpoint not found: {checkpoint}")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != CLAP_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"CLAP checkpoint digest mismatch: expected {CLAP_CHECKPOINT_SHA256}, "
            f"got {actual_sha256}."
        )
    return checkpoint


def load_candidate_audio(path: Path) -> tuple[Any, int]:
    try:
        from audiocraft.data.audio import audio_read
    except ImportError as error:
        raise RuntimeError(
            "CLAP scoring requires the AudioCraft environment from this "
            "repository's Docker image."
        ) from error
    try:
        waveform, sample_rate = audio_read(path)
    except Exception as error:
        raise RuntimeError(f"Unreadable audio {path}: {error}") from error
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise RuntimeError(f"Empty or malformed audio {path}.")
    if not waveform.isfinite().all():
        raise RuntimeError(f"Audio contains non-finite samples: {path}")
    return waveform, sample_rate


def fixed_duration_candidate_audio(path: Path) -> Any:
    try:
        import torch
        from audiocraft.data.audio_utils import convert_audio
    except ImportError as error:
        raise RuntimeError(
            "CLAP scoring requires torch and the AudioCraft audio utilities."
        ) from error
    waveform, sample_rate = load_candidate_audio(path)
    waveform = convert_audio(
        waveform.unsqueeze(0),
        from_rate=sample_rate,
        to_rate=CLAP_SAMPLE_RATE,
        to_channels=1,
    ).squeeze(0)
    minimum_frames = round(MIN_AUDIO_DURATION_SECONDS * CLAP_SAMPLE_RATE)
    if waveform.shape[-1] < minimum_frames:
        raise RuntimeError(
            f"Audio is shorter than {MIN_AUDIO_DURATION_SECONDS} second: {path}"
        )
    expected_frames = round(CLAP_DURATION_SECONDS * CLAP_SAMPLE_RATE)
    waveform = waveform[..., :expected_frames]
    if waveform.shape[-1] < expected_frames:
        waveform = torch.nn.functional.pad(
            waveform, (0, expected_frames - waveform.shape[-1])
        )
    return waveform


class ClapScorer:
    def __init__(self, checkpoint: Path, device: str) -> None:
        try:
            import laion_clap
            import torch
        except ImportError as error:
            raise RuntimeError(
                "CLAP scoring dependencies are missing; rebuild the repository "
                "Docker image."
            ) from error
        self.torch = torch
        self.device = select_device(device, torch)
        log_progress(f"loading CLAP music checkpoint on {self.device}")
        self.model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-base",
            device=self.device,
        )
        self.model.load_ckpt(str(checkpoint), verbose=False)
        self.model.eval()
        seed_everything(torch, CLAP_SCORING_SEED)

    def __enter__(self) -> "ClapScorer":
        return self

    def __exit__(self, *_args: Any) -> None:
        model = self.model
        self.model = None
        del model
        release_cuda(self.torch)

    def score_batch(
        self, records: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        torch = self.torch
        results: dict[str, dict[str, Any]] = {}
        valid_records: list[dict[str, Any]] = []
        waveforms: list[Any] = []
        for record in records:
            try:
                waveform = fixed_duration_candidate_audio(record["audio_path"])
            except (RuntimeError, OSError, ValueError) as error:
                results[record["id"]] = {
                    "status": "invalid",
                    "original_clap_score": None,
                    "effective_clap_score": None,
                    "rejection_reason": str(error),
                }
                continue
            valid_records.append(record)
            waveforms.append(waveform)
        if not valid_records:
            return results

        audio = torch.stack(waveforms).squeeze(1)
        with torch.inference_mode():
            audio_embeddings = self.model.get_audio_embedding_from_data(
                audio, use_tensor=True
            )
            audio_norms = torch.linalg.vector_norm(audio_embeddings, dim=1)
            if (
                not torch.isfinite(audio_embeddings).all()
                or not torch.isfinite(audio_norms).all()
                or torch.any(audio_norms <= 1e-8)
            ):
                raise RuntimeError("CLAP produced invalid audio embeddings.")
            original_embeddings = self.model.get_text_embedding(
                [record["original_caption"] for record in valid_records],
                use_tensor=True,
            )
            original_cosine = torch.nn.functional.cosine_similarity(
                audio_embeddings,
                original_embeddings,
                dim=1,
                eps=1e-8,
            )
            effective_cosine = original_cosine.clone()
            rewrite_indices = [
                index
                for index, record in enumerate(valid_records)
                if record["effective_caption"] != record["original_caption"]
            ]
            if rewrite_indices:
                rewrite_embeddings = self.model.get_text_embedding(
                    [
                        valid_records[index]["effective_caption"]
                        for index in rewrite_indices
                    ],
                    use_tensor=True,
                )
                rewrite_audio = audio_embeddings[rewrite_indices]
                rewrite_cosine = torch.nn.functional.cosine_similarity(
                    rewrite_audio,
                    rewrite_embeddings,
                    dim=1,
                    eps=1e-8,
                )
                effective_cosine[rewrite_indices] = rewrite_cosine

        originals = original_cosine.detach().cpu().tolist()
        effectives = effective_cosine.detach().cpu().tolist()
        for record, original_score, effective_score in zip(
            valid_records, originals, effectives
        ):
            if not math.isfinite(original_score) or not math.isfinite(effective_score):
                raise RuntimeError(
                    f"CLAP produced a non-finite score for {record['id']}."
                )
            results[record["id"]] = {
                "status": "eligible",
                "original_clap_score": float(original_score),
                "effective_clap_score": float(effective_score),
            }
        return results


def _validate_score_value(value: Any, label: str, track_id: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RuntimeError(f"Cached {label} is invalid for {track_id}.")
    return float(value)


def validate_score_record(
    record: dict[str, Any],
    candidate: dict[str, Any],
    expected_effective_caption: str,
) -> bool:
    track_id = candidate["id"]
    if record.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported cached score schema for {track_id}.")
    for field in ("id", "original_caption", "audio_sha256", "split"):
        expected = (
            candidate["original_caption"]
            if field == "original_caption"
            else candidate[field]
        )
        if record.get(field) != expected:
            raise RuntimeError(
                f"Cached score identity field {field} is invalid for {track_id}."
            )
    effective = record.get("effective_caption")
    if not isinstance(effective, str) or not effective.strip():
        raise RuntimeError(f"Cached effective caption is invalid for {track_id}.")
    if effective != expected_effective_caption:
        return False
    status = record.get("status")
    if status == "eligible":
        _validate_score_value(
            record.get("original_clap_score"), "original CLAP score", track_id
        )
        _validate_score_value(
            record.get("effective_clap_score"), "effective CLAP score", track_id
        )
    elif status == "invalid":
        if record.get("original_clap_score") is not None:
            raise RuntimeError(f"Invalid audio {track_id} has an original score.")
        if record.get("effective_clap_score") is not None:
            raise RuntimeError(f"Invalid audio {track_id} has an effective score.")
        if not isinstance(record.get("rejection_reason"), str):
            raise RuntimeError(f"Invalid audio {track_id} lacks a rejection reason.")
    else:
        raise RuntimeError(f"Cached score status is invalid for {track_id}.")
    return True


def _load_score_mapping(
    cache_dir: Path,
    candidates: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidates_by_id = {record["id"]: record for record in candidates}
    mapping: dict[str, dict[str, Any]] = {}
    final_path = cache_dir / "scores.jsonl"
    final_records = read_jsonl(final_path, missing_ok=True)
    for record in final_records:
        track_id = record.get("id")
        if track_id in mapping:
            raise RuntimeError(f"Duplicate cached score id in {final_path}: {track_id}")
        if track_id not in candidates_by_id:
            raise RuntimeError(f"Unknown cached score id in {final_path}: {track_id}")
        mapping[track_id] = record

    progress_path = cache_dir / "scores.progress.jsonl"
    for record in read_jsonl(
        progress_path,
        missing_ok=True,
        repair_truncated_tail=True,
    ):
        track_id = record.get("id")
        if track_id not in candidates_by_id:
            raise RuntimeError(f"Unknown cached score id in {progress_path}: {track_id}")
        mapping[track_id] = record

    reusable: dict[str, dict[str, Any]] = {}
    for track_id, record in mapping.items():
        candidate = candidates_by_id[track_id]
        caption = effective_caption(candidate, overrides.get(track_id))
        if validate_score_record(record, candidate, caption):
            reusable[track_id] = record
    return reusable


def _score_input(
    candidate: dict[str, Any],
    override: dict[str, Any] | None,
    cache_dir: Path,
) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "original_caption": candidate["original_caption"],
        "effective_caption": effective_caption(candidate, override),
        "audio_path": _candidate_audio_path(
            cache_dir, candidate["cached_audio_path"]
        ),
    }


def _score_record(
    candidate: dict[str, Any],
    caption: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "id": candidate["id"],
        "original_caption": candidate["original_caption"],
        "effective_caption": caption,
        "original_clap_score": result.get("original_clap_score"),
        "effective_clap_score": result.get("effective_clap_score"),
        "audio_sha256": candidate["audio_sha256"],
        "split": candidate["split"],
        "status": result.get("status"),
    }
    if result.get("status") == "invalid":
        record["rejection_reason"] = result.get("rejection_reason")
    validate_score_record(record, candidate, caption)
    return record


def score_candidates(
    candidates: list[dict[str, Any]],
    cache_dir: Path,
    overrides: dict[str, dict[str, Any]],
    batch_size: int,
    checkpoint_path: Path | None,
    device: str,
    *,
    score_batch: Callable[
        [list[dict[str, Any]]], dict[str, dict[str, Any]]
    ]
    | None = None,
) -> list[dict[str, Any]]:
    reusable = _load_score_mapping(cache_dir, candidates, overrides)
    progress_path = cache_dir / "scores.progress.jsonl"
    ordered_reusable = [
        reusable[candidate["id"]]
        for candidate in candidates
        if candidate["id"] in reusable
    ]
    write_jsonl(progress_path, ordered_reusable)
    pending = [
        candidate for candidate in candidates if candidate["id"] not in reusable
    ]
    if not pending:
        log_progress(f"reusing {len(candidates)} verified cached CLAP scores")
    scorer_context: Any = nullcontext()
    active_score_batch = score_batch
    if pending and active_score_batch is None:
        checkpoint = resolve_clap_checkpoint(checkpoint_path)
        scorer_context = ClapScorer(checkpoint, device)

    completed = len(candidates) - len(pending)
    with scorer_context as scorer:
        if active_score_batch is None and pending:
            active_score_batch = scorer.score_batch
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            log_progress(
                f"CLAP candidates {completed + start + 1}-"
                f"{completed + start + len(batch)} of {len(candidates)}"
            )
            inputs = [
                _score_input(candidate, overrides.get(candidate["id"]), cache_dir)
                for candidate in batch
            ]
            assert active_score_batch is not None
            results = active_score_batch(inputs)
            if set(results) != {candidate["id"] for candidate in batch}:
                raise RuntimeError(
                    "CLAP batch scorer returned incomplete or unexpected candidate IDs."
                )
            batch_records = []
            for candidate in batch:
                track_id = candidate["id"]
                caption = effective_caption(candidate, overrides.get(track_id))
                record = _score_record(candidate, caption, results[track_id])
                reusable[track_id] = record
                batch_records.append(record)
            append_jsonl(progress_path, batch_records)

    ordered = [reusable[candidate["id"]] for candidate in candidates]
    scores_path = cache_dir / "scores.jsonl"
    write_jsonl(scores_path, ordered)
    progress_path.unlink(missing_ok=True)
    return ordered


def load_frozen_evaluation_ids(path: Path = PROMPTS_PATH) -> set[str]:
    source_ids: set[str] = set()
    for line_number, record in enumerate(read_jsonl(path), start=1):
        source_id = record.get("source_id")
        if source_id is None or source_id == "":
            continue
        if not isinstance(source_id, str):
            raise RuntimeError(
                f"Frozen evaluation source_id at line {line_number} is invalid."
            )
        if source_id.strip():
            source_ids.add(source_id.strip())
    return source_ids


def _deduplication_winner(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return min(
        records,
        key=lambda record: (-record["effective_clap_score"], record["id"]),
    )


def build_selection(
    candidates: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
    frozen_evaluation_ids: set[str],
    train_count: int,
    valid_count: int,
) -> list[dict[str, Any]]:
    scores_by_id = {record["id"]: record for record in scores}
    candidate_ids = {record["id"] for record in candidates}
    if set(scores_by_id) != candidate_ids or len(scores) != len(candidates):
        raise RuntimeError("Every candidate must have exactly one score record.")
    selection: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for candidate in candidates:
        track_id = candidate["id"]
        score = scores_by_id.get(track_id)
        if score is None:
            raise RuntimeError(f"Missing score for candidate {track_id}.")
        override = overrides.get(track_id)
        record = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "id": track_id,
            "split": candidate["split"],
            "rank": None,
            "selected": False,
            "original_caption": candidate["original_caption"],
            "effective_caption": score["effective_caption"],
            "original_clap_score": score["original_clap_score"],
            "effective_clap_score": score["effective_clap_score"],
            "audio_sha256": candidate["audio_sha256"],
            "override": override,
            "decision": None,
            "rejection_reason": None,
            "retained_id": None,
            "caption_similarity": None,
        }
        if score["status"] == "invalid":
            record["decision"] = "invalid_audio"
            record["rejection_reason"] = score["rejection_reason"]
        elif track_id in frozen_evaluation_ids:
            record["decision"] = "excluded_frozen_evaluation"
            record["rejection_reason"] = "source_id appears in eval/prompts.jsonl"
        elif override is not None and override["action"] == "drop":
            record["decision"] = "dropped_by_override"
            record["rejection_reason"] = override.get(
                "reason", "manual drop override"
            )
        else:
            active.append(record)
        selection.append(record)

    by_audio_hash: dict[str, list[dict[str, Any]]] = {}
    for record in active:
        by_audio_hash.setdefault(record["audio_sha256"], []).append(record)
    after_audio: list[dict[str, Any]] = []
    for group in by_audio_hash.values():
        winner = _deduplication_winner(group)
        after_audio.append(winner)
        for record in group:
            if record is winner:
                continue
            record["decision"] = "duplicate_audio_sha256"
            record["rejection_reason"] = (
                f"audio SHA-256 duplicates retained candidate {winner['id']}"
            )
            record["retained_id"] = winner["id"]

    by_caption: dict[str, list[dict[str, Any]]] = {}
    for record in after_audio:
        by_caption.setdefault(
            normalize_caption(record["effective_caption"]), []
        ).append(record)
    deduplicated: list[dict[str, Any]] = []
    for group in by_caption.values():
        winner = _deduplication_winner(group)
        deduplicated.append(winner)
        for record in group:
            if record is winner:
                continue
            record["decision"] = "duplicate_normalized_caption"
            record["rejection_reason"] = (
                f"normalized caption duplicates retained candidate {winner['id']}"
            )
            record["retained_id"] = winner["id"]

    near_caption_rejected_ids: set[str] = set()
    for group in find_near_duplicate_caption_groups(deduplicated):
        winner = _deduplication_winner(group)
        for record in group:
            if record is winner:
                continue
            record["decision"] = "near_duplicate_caption"
            record["rejection_reason"] = (
                f"near-identical caption cluster retained candidate {winner['id']}"
            )
            record["retained_id"] = winner["id"]
            record["caption_similarity"] = near_caption_similarity(
                record["effective_caption"],
                winner["effective_caption"],
            )
            near_caption_rejected_ids.add(record["id"])
    deduplicated = [
        record
        for record in deduplicated
        if record["id"] not in near_caption_rejected_ids
    ]

    requested = {"train": train_count, "eval": valid_count}
    for split in SPLITS:
        ranked = sorted(
            (record for record in deduplicated if record["split"] == split),
            key=lambda record: (-record["effective_clap_score"], record["id"]),
        )
        for rank, record in enumerate(ranked, start=1):
            record["rank"] = rank
            if rank <= requested[split]:
                record["selected"] = True
                record["decision"] = "selected"
            else:
                record["decision"] = "below_selection_cutoff"
                record["rejection_reason"] = (
                    f"rank {rank} is below the top {requested[split]} {split} tracks"
                )
    return selection


def selected_counts(selection: list[dict[str, Any]]) -> dict[str, int]:
    return {
        split: sum(
            1
            for record in selection
            if record["selected"] and record["split"] == split
        )
        for split in SPLITS
    }


def require_exact_selection(
    selection: list[dict[str, Any]],
    train_count: int,
    valid_count: int,
) -> None:
    actual = selected_counts(selection)
    requested = {"train": train_count, "eval": valid_count}
    shortages = [
        f"{split}: requested {requested[split]}, eligible {actual[split]}"
        for split in SPLITS
        if actual[split] != requested[split]
    ]
    if shortages:
        raise RuntimeError(
            "Insufficient eligible candidates after exclusions and deduplication ("
            + "; ".join(shortages)
            + "). Increase --candidate-count, adjust overrides, or inspect "
            "selection.jsonl."
        )


def score_statistics(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "quantiles": {},
        }

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "quantiles": {
            "p25": quantile(0.25),
            "p50": quantile(0.50),
            "p75": quantile(0.75),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
        },
    }


def build_summary(
    candidates: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    config: dict[str, Any],
    train_count: int,
    valid_count: int,
    selection_sha256: str,
) -> dict[str, Any]:
    counts = selected_counts(selection)
    valid_scores = [
        record["effective_clap_score"]
        for record in selection
        if record["effective_clap_score"] is not None
    ]
    selected_scores = [
        record["effective_clap_score"]
        for record in selection
        if record["selected"]
    ]
    decisions: dict[str, int] = {}
    for record in selection:
        decisions[record["decision"]] = decisions.get(record["decision"], 0) + 1
    excluded_ids = sorted(
        record["id"]
        for record in selection
        if record["decision"] == "excluded_frozen_evaluation"
    )
    return {
        "schema_version": CURATION_SCHEMA_VERSION,
        "requested_counts": {
            "candidates": config["candidate_count"],
            "train": train_count,
            "valid": valid_count,
        },
        "actual_counts": {
            "candidates": len(candidates),
            "train": counts["train"],
            "valid": counts["eval"],
        },
        "candidate_counts_by_split": {
            split: sum(1 for record in candidates if record["split"] == split)
            for split in SPLITS
        },
        "selected_counts": {"train": counts["train"], "valid": counts["eval"]},
        "excluded_evaluation_count": len(excluded_ids),
        "excluded_evaluation_ids": excluded_ids,
        "dropped_count": decisions.get("dropped_by_override", 0),
        "rewritten_count": sum(
            1
            for record in selection
            if record["override"] is not None
            and record["override"]["action"] == "rewrite"
        ),
        "deduplicated_count": (
            decisions.get("duplicate_audio_sha256", 0)
            + decisions.get("duplicate_normalized_caption", 0)
            + decisions.get("near_duplicate_caption", 0)
        ),
        "invalid_count": decisions.get("invalid_audio", 0),
        "decision_counts": decisions,
        "scores": {
            "candidates": score_statistics(valid_scores),
            "selected": score_statistics(selected_scores),
        },
        "provenance": config,
        "selection_sha256": selection_sha256,
    }


def write_selection_artifacts(
    cache_dir: Path,
    candidates: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    config: dict[str, Any],
    train_count: int,
    valid_count: int,
) -> dict[str, Any]:
    selection_path = cache_dir / "selection.jsonl"
    log_progress(f"writing complete selection audit to {selection_path}")
    write_jsonl(selection_path, selection)
    digest = sha256_file(selection_path)
    summary = build_summary(
        candidates,
        selection,
        config,
        train_count,
        valid_count,
        digest,
    )
    summary_path = cache_dir / "summary.json"
    write_json(summary_path, summary)
    log_progress(
        f"wrote selection digest {digest} and summary to {summary_path}"
    )
    return summary


def music_metadata(
    track_id: str,
    prompt: str,
    extracted_keywords: list[tuple[str, float]],
) -> dict[str, Any]:
    prompt_lower = prompt.lower()
    instrument = next((name for name in INSTRUMENTS if name in prompt_lower), None)
    moods = [mood for mood in MOODS if mood in prompt_lower]
    return {
        "title": None,
        "artist": None,
        "key": None,
        "bpm": None,
        "genre": "lofi",
        "moods": moods,
        "keywords": [keyword for keyword, _score in extracted_keywords[:2]],
        "description": prompt,
        "name": track_id,
        "instrument": instrument,
    }


def _load_keyword_extractor() -> Callable[
    [list[str]], list[list[tuple[str, float]]]
]:
    try:
        from keybert import KeyBERT
    except ImportError as error:
        raise RuntimeError(
            "Dataset materialization requires KeyBERT from "
            "pipeline-requirements.txt."
        ) from error
    model = KeyBERT()
    return model.extract_keywords


def extract_selected_keywords(
    selected: list[dict[str, Any]],
    *,
    extractor: Callable[
        [list[str]], list[list[tuple[str, float]]]
    ]
    | None = None,
) -> dict[str, list[tuple[str, float]]]:
    if extractor is None:
        extractor = _load_keyword_extractor()
    keywords: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, len(selected), BATCH_SIZE):
        batch = selected[start : start + BATCH_SIZE]
        log_progress(
            f"KeyBERT selected tracks {start + 1}-"
            f"{start + len(batch)} of {len(selected)}"
        )
        prompts = [record["effective_caption"] for record in batch]
        result = extractor(prompts)
        if len(prompts) == 1 and (
            not result or isinstance(result[0], tuple)
        ):
            result = [result]
        if len(result) != len(batch):
            raise RuntimeError("KeyBERT returned an unexpected number of results.")
        for record, extracted in zip(batch, result):
            keywords[record["id"]] = extracted
    return keywords


def _replace_manifest_prefix(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in replacements:
            if value.startswith(old):
                return new + value[len(old) :]
        return value
    if isinstance(value, list):
        return [_replace_manifest_prefix(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_manifest_prefix(item, replacements)
            for key, item in value.items()
        }
    return value


def generate_staged_manifest(
    dataset_root: Path,
    split: str,
    manifest_path: Path,
    expected_tracks: int,
) -> None:
    log_progress(
        f"generating staged {split} manifest for {expected_tracks} tracks"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "audiocraft.data.audio_dataset",
            str((dataset_root / split).relative_to(AUDIOCRAFT_ROOT)),
            str(manifest_path.relative_to(AUDIOCRAFT_ROOT)),
        ],
        cwd=AUDIOCRAFT_ROOT,
        check=True,
    )
    records = read_jsonl(manifest_path)
    if len(records) != expected_tracks:
        raise RuntimeError(
            f"AudioCraft added {len(records)} of {expected_tracks} {split} tracks "
            "to the manifest. Check its preceding audio errors."
        )
    replacements = [
        (str(dataset_root.resolve()), str(DATASET_ROOT.resolve())),
        (
            dataset_root.relative_to(AUDIOCRAFT_ROOT).as_posix(),
            DATASET_ROOT.relative_to(AUDIOCRAFT_ROOT).as_posix(),
        ),
    ]
    records = [
        _replace_manifest_prefix(record, replacements) for record in records
    ]
    records.sort(
        key=lambda record: json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )
    write_jsonl(manifest_path, records)
    log_progress(
        f"validated staged {split} manifest with exactly "
        f"{expected_tracks} tracks"
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno == errno.ENOSPC:
            _raise_disk_error(error, destination)
        try:
            shutil.copy2(source, destination)
        except OSError as copy_error:
            _raise_disk_error(copy_error, destination)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _commit_staged_paths(replacements: list[tuple[Path, Path]]) -> None:
    token = uuid.uuid4().hex
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    log_progress("atomically installing the staged dataset, manifests, and config")
    try:
        for _staged, final in replacements:
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                backup = final.with_name(f".{final.name}.backup-{token}")
                final.replace(backup)
                backups.append((final, backup))
        for staged, final in replacements:
            staged.replace(final)
            installed.append(final)
    except Exception:
        log_progress("atomic installation failed; restoring previous final artifacts")
        for final in reversed(installed):
            if final.exists():
                _remove_path(final)
        for final, backup in reversed(backups):
            if backup.exists():
                backup.replace(final)
        raise
    for _final, backup in backups:
        if backup.exists():
            _remove_path(backup)
    log_progress("atomic installation completed")


def materialize_selection(
    selection: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    cache_dir: Path,
    train_count: int,
    valid_count: int,
    *,
    keyword_extractor: Callable[
        [list[str]], list[list[tuple[str, float]]]
    ]
    | None = None,
    manifest_generator: Callable[
        [Path, str, Path, int], None
    ] = generate_staged_manifest,
) -> None:
    require_exact_selection(selection, train_count, valid_count)
    candidates_by_id = {record["id"]: record for record in candidates}
    selected = sorted(
        (record for record in selection if record["selected"]),
        key=lambda record: (SPLITS.index(record["split"]), record["rank"], record["id"]),
    )
    log_progress(
        f"validating {len(selected)} selected cached audio files before staging"
    )
    for record in selected:
        candidate = candidates_by_id.get(record["id"])
        if candidate is None:
            raise RuntimeError(f"Selected candidate is missing: {record['id']}.")
        source = _candidate_audio_path(cache_dir, candidate["cached_audio_path"])
        if not source.is_file() or sha256_file(source) != record["audio_sha256"]:
            raise RuntimeError(
                f"Selected cached audio failed validation: {record['id']}."
            )

    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(
        _candidate_audio_path(
            cache_dir, candidates_by_id[record["id"]]["cached_audio_path"]
        ).stat().st_size
        for record in selected
    )
    if shutil.disk_usage(DATASET_ROOT.parent).free < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space to stage {required_bytes} bytes of selected audio."
        )

    stage_root = Path(
        tempfile.mkdtemp(prefix=".lofi-staging-", dir=DATASET_ROOT.parent)
    )
    log_progress(f"building the replacement dataset in {stage_root}")
    token = uuid.uuid4().hex
    manifest_paths = {
        split: AUDIOCRAFT_ROOT
        / "egs"
        / split
        / f".data.jsonl.staging-{token}"
        for split in SPLITS
    }
    config_final = AUDIOCRAFT_ROOT / "config" / "dset" / "audio" / "lofi.yaml"
    config_stage = config_final.with_name(f".lofi.yaml.staging-{token}")
    try:
        for split in SPLITS:
            (stage_root / split).mkdir(parents=True)
            manifest_paths[split].parent.mkdir(parents=True, exist_ok=True)
        config_stage.parent.mkdir(parents=True, exist_ok=True)
        config_stage.write_text(dataset_config_text(), encoding="utf-8")

        log_progress(f"extracting KeyBERT metadata for {len(selected)} selected tracks")
        keywords = extract_selected_keywords(
            selected, extractor=keyword_extractor
        )
        split_counts = {split: 0 for split in SPLITS}
        for index, record in enumerate(selected, start=1):
            candidate = candidates_by_id[record["id"]]
            source = _candidate_audio_path(
                cache_dir, candidate["cached_audio_path"]
            )
            stem = stage_root / record["split"] / filename_for_track(record["id"])
            _link_or_copy(source, Path(f"{stem}.mp3"))
            Path(f"{stem}.json").write_text(
                json.dumps(
                    music_metadata(
                        record["id"],
                        record["effective_caption"],
                        keywords[record["id"]],
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            split_counts[record["split"]] += 1
            if index % 100 == 0 or index == len(selected):
                log_progress(
                    f"staged {index}/{len(selected)} selected audio/metadata pairs"
                )

        expected = {"train": train_count, "eval": valid_count}
        if split_counts != expected:
            raise RuntimeError(
                f"Staged split counts are {split_counts}, expected {expected}."
            )
        log_progress(
            f"validated staged split counts: train={split_counts['train']}, "
            f"validation={split_counts['eval']}"
        )
        for split in SPLITS:
            manifest_generator(
                stage_root,
                split,
                manifest_paths[split],
                expected[split],
            )
            if len(read_jsonl(manifest_paths[split])) != expected[split]:
                raise RuntimeError(
                    f"Staged {split} manifest does not contain exactly "
                    f"{expected[split]} records."
                )

        _commit_staged_paths(
            [
                (stage_root, DATASET_ROOT),
                (
                    manifest_paths["train"],
                    AUDIOCRAFT_ROOT / "egs" / "train" / "data.jsonl",
                ),
                (
                    manifest_paths["eval"],
                    AUDIOCRAFT_ROOT / "egs" / "eval" / "data.jsonl",
                ),
                (config_stage, config_final),
            ]
        )
    finally:
        if stage_root.exists():
            log_progress(f"cleaning incomplete staging directory {stage_root}")
            shutil.rmtree(stage_root)
        for path in manifest_paths.values():
            path.unlink(missing_ok=True)
        config_stage.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    log_progress("stage 1/8: validating arguments, AudioCraft, and cache config")
    validate_args(args)
    check_audiocraft_checkout()
    cache_dir = args.cache_dir.expanduser().resolve()
    config = build_cache_config(args.candidate_count)
    ensure_cache_config(cache_dir, config)
    log_progress(f"using compatible curation cache {cache_dir}")

    log_progress("stage 2/8: collecting or validating candidate audio")
    candidates = collect_candidates(cache_dir, args.candidate_count)
    candidate_splits = {
        split: sum(1 for record in candidates if record["split"] == split)
        for split in SPLITS
    }
    log_progress(
        f"candidate pool ready: {len(candidates)} total, "
        f"train={candidate_splits['train']}, validation={candidate_splits['eval']}"
    )

    log_progress("stage 3/8: loading and validating manual overrides")
    overrides_path = resolve_overrides_path(args.overrides, cache_dir)
    overrides = load_overrides(overrides_path, candidates)
    override_counts = {
        action: sum(
            1 for override in overrides.values() if override["action"] == action
        )
        for action in ("drop", "rewrite")
    }
    if overrides_path is None:
        log_progress("no manual override file is active")
    else:
        log_progress(
            f"loaded {len(overrides)} overrides from {overrides_path}: "
            f"drops={override_counts['drop']}, rewrites={override_counts['rewrite']}"
        )

    log_progress("stage 4/8: scoring caption/audio pairs with CLAP")
    scores = score_candidates(
        candidates,
        cache_dir,
        overrides,
        args.clap_batch_size,
        args.clap_checkpoint,
        args.device,
    )
    valid_score_count = sum(
        1 for record in scores if record["status"] == "eligible"
    )
    log_progress(
        f"CLAP scoring ready: eligible={valid_score_count}, "
        f"invalid={len(scores) - valid_score_count}"
    )

    log_progress(
        "stage 5/8: excluding frozen evaluation IDs, deduplicating, and ranking"
    )
    frozen_ids = load_frozen_evaluation_ids()
    log_progress(f"loaded {len(frozen_ids)} frozen evaluation source IDs")
    selection = build_selection(
        candidates,
        scores,
        overrides,
        frozen_ids,
        args.train_count,
        args.valid_count,
    )
    decisions: dict[str, int] = {}
    for record in selection:
        decision = record["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
    counts = selected_counts(selection)
    log_progress(
        f"selection ready: train={counts['train']}/{args.train_count}, "
        f"validation={counts['eval']}/{args.valid_count}, "
        f"frozen_excluded={decisions.get('excluded_frozen_evaluation', 0)}, "
        f"dropped={decisions.get('dropped_by_override', 0)}, "
        "deduplicated="
        f"{decisions.get('duplicate_audio_sha256', 0) + decisions.get('duplicate_normalized_caption', 0) + decisions.get('near_duplicate_caption', 0)}, "
        f"near_caption_duplicates={decisions.get('near_duplicate_caption', 0)}, "
        f"invalid={decisions.get('invalid_audio', 0)}"
    )

    log_progress("stage 6/8: writing reproducibility and selection audit artifacts")
    summary = write_selection_artifacts(
        cache_dir,
        candidates,
        selection,
        config,
        args.train_count,
        args.valid_count,
    )
    require_exact_selection(selection, args.train_count, args.valid_count)

    log_progress("stage 7/8: staging and validating the selected AudioCraft dataset")
    materialize_selection(
        selection,
        candidates,
        cache_dir,
        args.train_count,
        args.valid_count,
    )
    log_progress(
        "stage 8/8: preparation complete; "
        f"prepared exactly {args.train_count} train and "
        f"{args.valid_count} validation tracks"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f"[prepare] failed: {error}") from error
