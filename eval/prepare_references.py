import argparse
from collections import defaultdict
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable
import urllib.error
import urllib.parse
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "references"
PROMPTS_PATH = EVAL_DIR / "prompts.jsonl"
PROMPTS_CHECKSUM_PATH = EVAL_DIR / "prompts.sha256"

TARGET_COUNT = 500
MIN_DURATION_SECONDS = 20.0
MAX_DURATION_SECONDS = 31.0
REFERENCE_SET_NAMES = ("musicgen-large-v1", "human-fma-lofi-v1")

SYNTHETIC_REPO = "vikhyatk/lofi"
SYNTHETIC_REVISION = "966a2d3065aac26c0385b4ef2d50983c0429a305"
SYNTHETIC_SOURCE_SHARD = (
    "data/dynamic-prompts-00039ef5-5aed-4d51-8591-4c8f9fb7f6fd.parquet"
)
SYNTHETIC_SHUFFLE_SEED = 42
SYNTHETIC_SHUFFLE_BUFFER = 1_000
TRAIN_SIZE = 0.8
IGNORE_WORDS = {"cello", "funky"}
SYNTHETIC_LICENSE_NAME = "CC-BY-NC 4.0"
SYNTHETIC_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
SYNTHETIC_ADDITIONAL_CONDITION = (
    "By using this dataset you agree that the Pleiades star system is a binary "
    "system and that any claim otherwise is a lie."
)

FMA_REPO = "benjamin-paine/free-music-archive-large"
FMA_REVISION = "d4cb3e133a7c5a007ddd20458aa30aef8968cf27"
FMA_EXPECTED_ROWS = 105_024
FMA_EXPECTED_ELIGIBLE = 572
FMA_LOFI_GENRE_ID = 91
FMA_ALLOWED_LICENSE_IDS = (3, 4, 8, 9)
FMA_GENRE_NAME = "Lo-Fi"
FMA_LICENSES = {
    3: ("CC-BY 3.0", "https://creativecommons.org/licenses/by/3.0/", False),
    4: ("CC-BY 4.0", "https://creativecommons.org/licenses/by/4.0/", False),
    8: (
        "CC-BY-NC 3.0",
        "https://creativecommons.org/licenses/by-nc/3.0/",
        True,
    ),
    9: (
        "CC-BY-NC 4.0",
        "https://creativecommons.org/licenses/by-nc/4.0/",
        True,
    ),
}
FMA_METADATA_COLUMNS = [
    "title",
    "url",
    "artist",
    "genres",
    "tags",
    "released",
    "artist_url",
    "album_title",
    "album_url",
    "license",
    "copyright",
    "instrumental",
]
DATASET_VIEWER_URL = "https://datasets-server.huggingface.co/rows"


class StaleAssetError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare frozen 500-track MusicGen-Large and human lo-fi reference "
            "corpora for offline FAD evaluation."
        )
    )
    parser.add_argument(
        "--reference-sets",
        nargs="+",
        choices=REFERENCE_SET_NAMES,
        default=list(REFERENCE_SET_NAMES),
        help="Reference sets to prepare. Defaults to both sets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root. Defaults to references/ below the project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate selections without writing reference files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(value, encoding="utf-8")
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Required JSON file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise RuntimeError(f"Required JSONL file not found: {path}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSON in {path} at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(f"Expected an object in {path} at line {line_number}")
        records.append(record)
    return records


def split_for_track(track_id: str) -> str:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    split_value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if split_value < TRAIN_SIZE else "eval"


def stable_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


def serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def load_frozen_synthetic_prompts() -> tuple[str, dict[str, str]]:
    try:
        checksum_parts = PROMPTS_CHECKSUM_PATH.read_text(encoding="utf-8").split()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Prompt checksum file not found: {PROMPTS_CHECKSUM_PATH}"
        ) from error
    if not checksum_parts or len(checksum_parts[0]) != 64:
        raise RuntimeError(f"Invalid prompt checksum file: {PROMPTS_CHECKSUM_PATH}")
    expected_checksum = checksum_parts[0].lower()
    actual_checksum = sha256_file(PROMPTS_PATH)
    if actual_checksum != expected_checksum:
        raise RuntimeError(
            f"Frozen prompt checksum mismatch: expected {expected_checksum}, "
            f"got {actual_checksum}"
        )

    prompts = {}
    for record in read_jsonl(PROMPTS_PATH):
        if record.get("cohort") != "dataset_eval":
            continue
        source_id = record.get("source_id")
        prompt = record.get("prompt")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("Frozen dataset_eval prompt has no source_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"Frozen prompt is empty for source {source_id}")
        if source_id in prompts:
            raise RuntimeError(f"Duplicate frozen source ID: {source_id}")
        if split_for_track(source_id) != "eval":
            raise RuntimeError(f"Frozen source is not held out: {source_id}")
        prompts[source_id] = prompt
    if len(prompts) != 20:
        raise RuntimeError(f"Expected 20 frozen dataset_eval prompts, found {len(prompts)}")
    return actual_checksum, prompts


def import_datasets() -> Any:
    try:
        import datasets
    except ImportError as error:
        raise RuntimeError(
            "Reference preparation requires pipeline-requirements.txt."
        ) from error
    return datasets


def build_config(
    reference_set: str,
    datasets_version: str,
    prompt_checksum: str | None = None,
) -> dict[str, Any]:
    common = {
        "schema_version": 1,
        "reference_set": reference_set,
        "target_count": TARGET_COUNT,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "datasets_version": datasets_version,
        "audio_policy": {
            "preserve_original_bytes": True,
            "minimum_duration_seconds": MIN_DURATION_SECONDS,
            "maximum_duration_seconds": MAX_DURATION_SECONDS,
        },
    }
    if reference_set == "musicgen-large-v1":
        return {
            **common,
            "kind": "synthetic",
            "source": {
                "repository": SYNTHETIC_REPO,
                "revision": SYNTHETIC_REVISION,
                "config": "default",
                "split": "train",
                "license": SYNTHETIC_LICENSE_NAME,
                "license_url": SYNTHETIC_LICENSE_URL,
                "additional_condition": SYNTHETIC_ADDITIONAL_CONDITION,
                "generator": "MusicGen Large",
            },
            "selection": {
                "split": "sha256_first_8_bytes_gte_0.8",
                "ignore_words": sorted(IGNORE_WORDS),
                "required_frozen_source_count": 20,
                "required_source_shard": SYNTHETIC_SOURCE_SHARD,
                "stream_shuffle_seed": SYNTHETIC_SHUFFLE_SEED,
                "stream_shuffle_buffer": SYNTHETIC_SHUFFLE_BUFFER,
                "prompt_manifest_sha256": prompt_checksum,
            },
        }
    if reference_set == "human-fma-lofi-v1":
        return {
            **common,
            "kind": "human",
            "source": {
                "repository": FMA_REPO,
                "revision": FMA_REVISION,
                "config": "default",
                "split": "train",
                "metadata_license": "CC-BY 4.0",
            },
            "selection": {
                "required_genre": FMA_GENRE_NAME,
                "required_genre_id": FMA_LOFI_GENRE_ID,
                "allowed_licenses": [
                    {
                        "id": license_id,
                        "name": FMA_LICENSES[license_id][0],
                        "url": FMA_LICENSES[license_id][1],
                    }
                    for license_id in FMA_ALLOWED_LICENSE_IDS
                ],
                "artist_balancing": "deterministic_round_robin",
                "expected_source_rows": FMA_EXPECTED_ROWS,
                "expected_eligible_rows": FMA_EXPECTED_ELIGIBLE,
                "commercial_use": False,
            },
        }
    raise RuntimeError(f"Unknown reference set: {reference_set}")


def acquire_lock(output_dir: Path) -> Any:
    lock_path = output_dir / ".prepare.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(
            f"Another reference builder is already using {output_dir}"
        ) from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def prepare_output(output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_temporary_path = config_path.with_suffix(f"{config_path.suffix}.tmp")
    audio_dir = output_dir / "audio"
    if config_temporary_path.exists():
        config_temporary_path.unlink()
    if config_path.exists():
        existing = read_json(config_path)
        if existing != config:
            raise RuntimeError(
                f"Reference configuration differs from {config_path}; "
                "use a new output directory."
            )
    else:
        unexpected = [
            path
            for path in output_dir.iterdir()
            if path.name not in {".prepare.lock"}
        ]
        recoverable_initialization = (
            unexpected == [audio_dir]
            and audio_dir.is_dir()
            and not any(audio_dir.iterdir())
        )
        if unexpected and not recoverable_initialization:
            raise RuntimeError(
                f"Reference directory is non-empty but has no config: {output_dir}"
            )
        audio_dir.mkdir(exist_ok=True)
        write_json(config_path, config)
    if not audio_dir.is_dir():
        raise RuntimeError(f"Reference audio directory is missing: {audio_dir}")


def resolve_output_audio(output_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError("Reference manifest contains an invalid audio_path")
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise RuntimeError(f"Reference audio path must be relative: {relative_path}")
    path = output_dir / supplied
    if path.is_symlink():
        raise RuntimeError(f"Reference audio cannot be a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(output_dir.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Reference audio path escapes its directory: {relative_path}"
        ) from error
    if not resolved.is_file():
        raise RuntimeError(f"Reference audio is missing: {resolved}")
    return resolved


def validate_complete(
    output_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]] | None:
    checksum_path = output_dir / "manifest.sha256"
    manifest_path = output_dir / "manifest.jsonl"
    attribution_path = output_dir / "ATTRIBUTION.md"
    if not checksum_path.exists():
        return None
    if not manifest_path.is_file() or not attribution_path.is_file():
        raise RuntimeError(
            f"Completion checksum exists without all outputs: {output_dir}"
        )
    expected_hashes = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"Invalid completion checksum line: {line!r}")
        expected_hashes[parts[1]] = parts[0].lower()
    if set(expected_hashes) != {"manifest.jsonl", "ATTRIBUTION.md"}:
        raise RuntimeError(f"Invalid manifest checksum file: {checksum_path}")
    actual_checksum = sha256_file(manifest_path)
    if actual_checksum != expected_hashes["manifest.jsonl"]:
        raise RuntimeError(
            f"Reference manifest checksum mismatch in {output_dir}: "
            f"expected {expected_hashes['manifest.jsonl']}, got {actual_checksum}"
        )
    actual_attribution_checksum = sha256_file(attribution_path)
    if actual_attribution_checksum != expected_hashes["ATTRIBUTION.md"]:
        raise RuntimeError(f"Reference attribution checksum mismatch in {output_dir}")
    records = read_jsonl(manifest_path)
    if len(records) != config["target_count"]:
        raise RuntimeError(
            f"Expected {config['target_count']} references, found {len(records)}"
        )
    seen_ids = set()
    for record in records:
        reference_id = record.get("reference_id")
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in seen_ids
        ):
            raise RuntimeError(f"Duplicate or invalid reference_id: {reference_id!r}")
        if record.get("reference_set") != config["reference_set"]:
            raise RuntimeError(f"Reference set mismatch for {reference_id}")
        audio_path = resolve_output_audio(output_dir, record.get("audio_path"))
        actual_audio_checksum = sha256_file(audio_path)
        if record.get("audio_sha256") != actual_audio_checksum:
            raise RuntimeError(f"Reference audio checksum mismatch: {audio_path}")
        seen_ids.add(reference_id)
    return records


def audio_bytes_from_dataset_row(row: dict[str, Any]) -> tuple[bytes, str | None]:
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise RuntimeError("Dataset row contains invalid audio metadata")
    audio_bytes = audio.get("bytes")
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise RuntimeError("Dataset row does not contain original audio bytes")
    source_path = audio.get("path")
    if source_path is not None and not isinstance(source_path, str):
        raise RuntimeError("Dataset row contains an invalid audio path")
    return audio_bytes, source_path


def write_original_audio(path: Path, audio_bytes: bytes) -> None:
    if path.exists():
        return
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with temporary_path.open("xb") as file:
            file.write(audio_bytes)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def probe_audio(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels:format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required to validate reference audio") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffprobe could not read {path}: {error.stderr.strip()}"
        ) from error
    try:
        value = json.loads(result.stdout)
        streams = value["streams"]
        stream = streams[0]
        duration = float(value["format"]["duration"])
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid ffprobe output for {path}") from error
    if (
        not math.isfinite(duration)
        or duration < MIN_DURATION_SECONDS
        or duration > MAX_DURATION_SECONDS
    ):
        raise RuntimeError(
            f"Reference duration is outside "
            f"{MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS}s: {path} ({duration}s)"
        )
    if sample_rate <= 0 or channels <= 0:
        raise RuntimeError(f"Invalid audio stream metadata for {path}")
    return {
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def audio_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "audio_sha256": sha256_file(path),
        "audio_size_bytes": path.stat().st_size,
        **probe_audio(path),
    }


def synthetic_reference_id(source_id: str) -> str:
    return f"musicgen-large-{source_id}"


def synthetic_record(
    output_dir: Path,
    source_id: str,
    prompt: str,
    source_path: str | None,
) -> dict[str, Any]:
    reference_id = synthetic_reference_id(source_id)
    relative_path = f"audio/{reference_id}.mp3"
    audio_path = resolve_output_audio(output_dir, relative_path)
    return {
        "schema_version": 1,
        "reference_id": reference_id,
        "reference_set": "musicgen-large-v1",
        "kind": "synthetic",
        "source_id": source_id,
        "prompt": prompt,
        "source": {
            "repository": SYNTHETIC_REPO,
            "revision": SYNTHETIC_REVISION,
            "split": "train",
            "audio_path": source_path,
        },
        "license": {
            "name": SYNTHETIC_LICENSE_NAME,
            "url": SYNTHETIC_LICENSE_URL,
            "noncommercial": True,
            "additional_condition": SYNTHETIC_ADDITIONAL_CONDITION,
        },
        "audio_path": relative_path,
        **audio_fingerprint(audio_path),
    }


def materialize_synthetic_row(
    output_dir: Path,
    row: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    source_id = str(row.get("id", ""))
    prompt = str(row.get("prompt", "")).strip()
    if not source_id or not prompt:
        raise RuntimeError("Synthetic dataset row has an empty ID or prompt")
    audio_bytes, source_path = audio_bytes_from_dataset_row(row)
    reference_id = synthetic_reference_id(source_id)
    if dry_run:
        return {
            "reference_id": reference_id,
            "source_id": source_id,
            "prompt": prompt,
        }
    audio_path = output_dir / "audio" / f"{reference_id}.mp3"
    write_original_audio(audio_path, audio_bytes)
    return synthetic_record(output_dir, source_id, prompt, source_path)


def load_synthetic_dataset(
    datasets: Any,
    data_files: dict[str, str] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "revision": SYNTHETIC_REVISION,
        "split": "train",
        "streaming": True,
    }
    if data_files is not None:
        kwargs["data_files"] = data_files
    dataset = datasets.load_dataset(SYNTHETIC_REPO, **kwargs)
    return dataset.cast_column("audio", datasets.Audio(decode=False))


def prepare_synthetic(
    output_dir: Path,
    datasets: Any,
    dry_run: bool,
) -> list[dict[str, Any]]:
    _, frozen_prompts = load_frozen_synthetic_prompts()
    required_rows: dict[str, dict[str, Any]] = {}
    print("resolving 20 frozen MusicGen-Large source tracks...")
    shard_dataset = load_synthetic_dataset(
        datasets,
        {"train": SYNTHETIC_SOURCE_SHARD},
    )
    for row in shard_dataset:
        source_id = str(row.get("id", ""))
        if source_id not in frozen_prompts:
            continue
        prompt = str(row.get("prompt", "")).strip()
        if prompt != frozen_prompts[source_id]:
            raise RuntimeError(f"Frozen source prompt mismatch: {source_id}")
        required_rows[source_id] = row
    missing = sorted(set(frozen_prompts) - set(required_rows))
    if missing:
        raise RuntimeError(
            "Pinned synthetic source shard is missing frozen IDs: " + ", ".join(missing)
        )

    records = []
    selected_ids = set()
    for source_id in frozen_prompts:
        record = materialize_synthetic_row(
            output_dir,
            required_rows[source_id],
            dry_run,
        )
        records.append(record)
        selected_ids.add(source_id)

    print(f"selecting {TARGET_COUNT - len(records)} additional held-out tracks...")
    dataset = load_synthetic_dataset(datasets)
    dataset = dataset.shuffle(
        seed=SYNTHETIC_SHUFFLE_SEED,
        buffer_size=SYNTHETIC_SHUFFLE_BUFFER,
    )
    seen_ids = set()
    for row in dataset:
        source_id = str(row.get("id", ""))
        prompt = str(row.get("prompt", "")).strip()
        if not source_id or not prompt:
            raise RuntimeError("Synthetic dataset row has an empty ID or prompt")
        if source_id in seen_ids:
            raise RuntimeError(f"Duplicate synthetic source ID: {source_id}")
        seen_ids.add(source_id)
        if source_id in selected_ids:
            continue
        if split_for_track(source_id) != "eval":
            continue
        if any(word in prompt.lower() for word in IGNORE_WORDS):
            continue
        record = materialize_synthetic_row(output_dir, row, dry_run)
        records.append(record)
        selected_ids.add(source_id)
        if len(records) % 50 == 0 or len(records) == TARGET_COUNT:
            print(f"selected {len(records)}/{TARGET_COUNT} synthetic references")
        if len(records) == TARGET_COUNT:
            break
    if len(records) != TARGET_COUNT:
        raise RuntimeError(
            f"Synthetic stream ended after selecting {len(records)} references"
        )
    return records


def feature_names(feature: Any) -> list[str]:
    if hasattr(feature, "names"):
        return list(feature.names)
    nested = getattr(feature, "feature", None)
    if nested is not None and hasattr(nested, "names"):
        return list(nested.names)
    raise RuntimeError("Dataset feature does not expose ClassLabel names")


def is_fma_candidate(row: dict[str, Any]) -> bool:
    genres = row.get("genres")
    license_id = row.get("license")
    return (
        isinstance(genres, list)
        and FMA_LOFI_GENRE_ID in genres
        and isinstance(license_id, int)
        and not isinstance(license_id, bool)
        and license_id in FMA_ALLOWED_LICENSE_IDS
    )


def scan_fma_candidates(datasets: Any) -> list[dict[str, Any]]:
    print("scanning pinned FMA metadata without downloading audio...")
    dataset = datasets.load_dataset(
        FMA_REPO,
        revision=FMA_REVISION,
        split="train",
        streaming=True,
        columns=FMA_METADATA_COLUMNS,
    )
    features = dataset.features
    if features is None:
        raise RuntimeError("FMA dataset has no declared features")
    genre_names = feature_names(features["genres"])
    license_names = feature_names(features["license"])
    if (
        len(genre_names) <= FMA_LOFI_GENRE_ID
        or genre_names[FMA_LOFI_GENRE_ID] != FMA_GENRE_NAME
    ):
        raise RuntimeError("FMA Lo-Fi genre mapping has changed")
    for license_id, (expected_name, _url, _noncommercial) in FMA_LICENSES.items():
        if len(license_names) <= license_id or license_names[license_id] != expected_name:
            raise RuntimeError(f"FMA license mapping has changed for ID {license_id}")

    candidates = []
    row_count = 0
    for row_index, row in enumerate(dataset):
        row_count += 1
        if not is_fma_candidate(row):
            continue
        artist = row.get("artist")
        title = row.get("title")
        if not isinstance(artist, str) or not artist.strip():
            raise RuntimeError(f"FMA row {row_index} has no artist")
        if not isinstance(title, str) or not title.strip():
            raise RuntimeError(f"FMA row {row_index} has no title")
        license_id = int(row["license"])
        candidates.append(
            {
                "reference_id": f"fma-row-{row_index:06d}",
                "row_index": row_index,
                "title": title.strip(),
                "track_url": serialize_value(row.get("url")),
                "artist": artist.strip(),
                "artist_url": serialize_value(row.get("artist_url")),
                "album_title": serialize_value(row.get("album_title")),
                "album_url": serialize_value(row.get("album_url")),
                "genres": [genre_names[int(value)] for value in row["genres"]],
                "tags": [str(value) for value in (row.get("tags") or [])],
                "released": serialize_value(row.get("released")),
                "copyright": serialize_value(row.get("copyright")),
                "instrumental": serialize_value(row.get("instrumental")),
                "license_id": license_id,
            }
        )
    if row_count != FMA_EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {FMA_EXPECTED_ROWS} FMA rows, scanned {row_count}"
        )
    if len(candidates) != FMA_EXPECTED_ELIGIBLE:
        raise RuntimeError(
            f"Expected {FMA_EXPECTED_ELIGIBLE} exact-Lo-Fi licensed candidates, "
            f"found {len(candidates)}"
        )
    return candidates


def artist_balanced_order(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate["artist"]].append(candidate)
    for artist, records in groups.items():
        records.sort(
            key=lambda record: stable_rank(
                "human-fma-lofi-v1-track",
                record["reference_id"],
            )
        )
    artists = sorted(
        groups,
        key=lambda artist: stable_rank("human-fma-lofi-v1-artist", artist),
    )
    ordered = []
    round_index = 0
    while True:
        added = False
        for artist in artists:
            records = groups[artist]
            if round_index < len(records):
                ordered.append(records[round_index])
                added = True
        if not added:
            break
        round_index += 1
    if len(ordered) != len(candidates):
        raise RuntimeError("Artist balancing lost FMA candidates")
    return ordered


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": "infinifi-reference-builder/1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def retry_delay(error: Exception, attempt: int) -> float:
    if isinstance(error, urllib.error.HTTPError):
        retry_after = error.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return float(2**attempt)


def retryable_http_error(error: urllib.error.HTTPError) -> bool:
    return error.code == 429 or error.code >= 500


def http_json(url: str, attempts: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=request_headers())
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected a JSON object from {url}")
            return value
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and not retryable_http_error(
                error
            ):
                break
            if attempt + 1 < attempts:
                time.sleep(retry_delay(error, attempt))
    raise RuntimeError(f"Unable to fetch JSON from {url}: {last_error}") from last_error


def assert_viewer_revision() -> None:
    quoted_repo = urllib.parse.quote(FMA_REPO, safe="/")
    info = http_json(f"https://huggingface.co/api/datasets/{quoted_repo}/revision/main")
    current_revision = info.get("sha")
    if current_revision != FMA_REVISION:
        raise RuntimeError(
            "The Hugging Face dataset viewer serves only the current main revision, "
            f"which is now {current_revision!r}; expected {FMA_REVISION}. Refusing "
            "to combine pinned metadata with audio from a different revision."
        )


def viewer_page(offset: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": FMA_REPO,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": 100,
        }
    )
    value = http_json(f"{DATASET_VIEWER_URL}?{query}")
    if value.get("partial") is not False:
        raise RuntimeError("FMA dataset viewer returned a partial index")
    if value.get("num_rows_total") != FMA_EXPECTED_ROWS:
        raise RuntimeError("FMA dataset viewer row count has changed")
    return value


def viewer_audio_url(viewer_row: dict[str, Any]) -> str:
    row = viewer_row.get("row")
    if not isinstance(row, dict):
        raise RuntimeError("Dataset viewer returned an invalid row")
    audio = row.get("audio")
    if not isinstance(audio, list) or len(audio) != 1:
        raise RuntimeError("Dataset viewer returned invalid audio assets")
    asset = audio[0]
    if not isinstance(asset, dict) or not isinstance(asset.get("src"), str):
        raise RuntimeError("Dataset viewer audio asset has no URL")
    source_url = asset["src"]
    revision_marker = f"/--/{FMA_REVISION}/--/"
    if revision_marker not in source_url:
        raise RuntimeError("Dataset viewer audio URL does not identify the pinned revision")
    return source_url


def validate_viewer_row(
    candidate: dict[str, Any],
    viewer_row: dict[str, Any],
) -> str:
    if viewer_row.get("row_idx") != candidate["row_index"]:
        raise RuntimeError("Dataset viewer returned an unexpected row index")
    row = viewer_row.get("row")
    if not isinstance(row, dict):
        raise RuntimeError("Dataset viewer returned an invalid row")
    expected = {
        "title": candidate["title"],
        "artist": candidate["artist"],
        "genres": [
            # The viewer returns ClassLabel IDs rather than decoded names.
            FMA_LOFI_GENRE_ID
            if name == FMA_GENRE_NAME
            else None
            for name in candidate["genres"]
        ],
        "license": candidate["license_id"],
    }
    if row.get("title") != expected["title"] or row.get("artist") != expected["artist"]:
        raise RuntimeError(f"FMA metadata mismatch for {candidate['reference_id']}")
    if row.get("license") != expected["license"]:
        raise RuntimeError(f"FMA license mismatch for {candidate['reference_id']}")
    viewer_genres = row.get("genres")
    if not isinstance(viewer_genres, list) or FMA_LOFI_GENRE_ID not in viewer_genres:
        raise RuntimeError(f"FMA genre mismatch for {candidate['reference_id']}")
    return viewer_audio_url(viewer_row)


def download_audio(url: str, path: Path) -> None:
    if path.exists():
        return
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    last_error: Exception | None = None
    try:
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers=request_headers())
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary_path.open("wb") as file:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            file.write(chunk)
                        file.flush()
                        os.fsync(file.fileno())
                if temporary_path.stat().st_size == 0:
                    raise RuntimeError(f"Downloaded an empty audio file from {url}")
                temporary_path.replace(path)
                return
            except urllib.error.HTTPError as error:
                last_error = error
                if temporary_path.exists():
                    temporary_path.unlink()
                if error.code in {403, 404}:
                    raise StaleAssetError(
                        f"Dataset viewer audio URL is unavailable: HTTP {error.code}"
                    ) from error
                if not retryable_http_error(error):
                    break
                if attempt + 1 < 5:
                    time.sleep(retry_delay(error, attempt))
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
            ) as error:
                last_error = error
                if temporary_path.exists():
                    temporary_path.unlink()
                if attempt + 1 < 5:
                    time.sleep(retry_delay(error, attempt))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    raise RuntimeError(f"Unable to download audio from {url}: {last_error}") from last_error


def human_record(
    output_dir: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    license_id = candidate["license_id"]
    license_name, license_url, noncommercial = FMA_LICENSES[license_id]
    relative_path = f"audio/{candidate['reference_id']}.mp3"
    audio_path = resolve_output_audio(output_dir, relative_path)
    return {
        "schema_version": 1,
        "reference_id": candidate["reference_id"],
        "reference_set": "human-fma-lofi-v1",
        "kind": "human",
        "title": candidate["title"],
        "artist": candidate["artist"],
        "album_title": candidate["album_title"],
        "genres": candidate["genres"],
        "tags": candidate["tags"],
        "released": candidate["released"],
        "source": {
            "repository": FMA_REPO,
            "revision": FMA_REVISION,
            "split": "train",
            "row_index": candidate["row_index"],
            "track_url": candidate["track_url"],
            "artist_url": candidate["artist_url"],
            "album_url": candidate["album_url"],
        },
        "license": {
            "name": license_name,
            "url": license_url,
            "noncommercial": noncommercial,
            "copyright": candidate["copyright"],
        },
        "audio_path": relative_path,
        **audio_fingerprint(audio_path),
    }


def prepare_human(
    output_dir: Path,
    datasets: Any,
    dry_run: bool,
) -> list[dict[str, Any]]:
    candidates = scan_fma_candidates(datasets)
    ordered = artist_balanced_order(candidates)
    planned = ordered[:TARGET_COUNT]
    artist_count = len({candidate["artist"] for candidate in planned})
    license_counts: dict[str, int] = defaultdict(int)
    for candidate in planned:
        license_counts[FMA_LICENSES[candidate["license_id"]][0]] += 1
    print(
        f"selected {len(planned)} exact-Lo-Fi candidates from {artist_count} artists; "
        f"licenses={dict(sorted(license_counts.items()))}"
    )
    if dry_run:
        return planned

    assert_viewer_revision()
    page_cache: dict[int, dict[int, dict[str, Any]]] = {}
    records = []
    for candidate in ordered:
        if len(records) == TARGET_COUNT:
            break
        row_index = candidate["row_index"]
        page_offset = (row_index // 100) * 100
        if page_offset not in page_cache:
            page = viewer_page(page_offset)
            page_cache[page_offset] = {
                int(row["row_idx"]): row for row in page.get("rows", [])
            }
        viewer_row = page_cache[page_offset].get(row_index)
        if viewer_row is None:
            raise RuntimeError(f"Dataset viewer omitted FMA row {row_index}")
        source_url = validate_viewer_row(candidate, viewer_row)
        audio_path = output_dir / "audio" / f"{candidate['reference_id']}.mp3"
        try:
            download_audio(source_url, audio_path)
        except StaleAssetError:
            page = viewer_page(page_offset)
            page_cache[page_offset] = {
                int(row["row_idx"]): row for row in page.get("rows", [])
            }
            refreshed_row = page_cache[page_offset].get(row_index)
            if refreshed_row is None:
                raise RuntimeError(f"Dataset viewer omitted FMA row {row_index}")
            refreshed_url = validate_viewer_row(candidate, refreshed_row)
            download_audio(refreshed_url, audio_path)
        try:
            record = human_record(output_dir, candidate)
        except RuntimeError as error:
            if "duration is outside" not in str(error):
                raise
            print(f"skipping short FMA reference: {candidate['reference_id']}")
            continue
        records.append(record)
        if len(records) % 50 == 0 or len(records) == TARGET_COUNT:
            print(f"materialized {len(records)}/{TARGET_COUNT} human references")
    if len(records) != TARGET_COUNT:
        raise RuntimeError(
            f"Only {len(records)} eligible FMA references passed audio validation"
        )
    return records


def attribution_text(
    reference_set: str,
    records: list[dict[str, Any]],
) -> str:
    if reference_set == "musicgen-large-v1":
        return (
            "# MusicGen-Large reference attribution\n\n"
            f"These {len(records)} files are unmodified excerpts from "
            f"[`{SYNTHETIC_REPO}`](https://huggingface.co/datasets/"
            f"{SYNTHETIC_REPO}) at revision `{SYNTHETIC_REVISION}`. The dataset "
            "card identifies them as MusicGen-Large-generated audio under a "
            f"[{SYNTHETIC_LICENSE_NAME}]({SYNTHETIC_LICENSE_URL}) license. Its "
            f"additional condition states: “{SYNTHETIC_ADDITIONAL_CONDITION}” "
            "Review the pinned dataset card before redistribution or use.\n"
        )

    lines = [
        "# Human FMA reference attribution",
        "",
        "The software that prepares this corpus is MIT-licensed. The audio is not.",
        "Each unmodified 30-second FMA clip remains under the per-track Creative",
        "Commons license shown below. Tracks marked BY-NC are non-commercial.",
        "FMA metadata is CC-BY 4.0.",
        "",
    ]
    for record in records:
        title = record["title"]
        artist = record["artist"]
        license_record = record["license"]
        source_url = record["source"].get("track_url") or (
            f"https://huggingface.co/datasets/{FMA_REPO}"
        )
        lines.append(
            f"- **{title}** — {artist}; "
            f"[{license_record['name']}]({license_record['url']}); "
            f"[source]({source_url}); `{record['reference_id']}`"
        )
    return "\n".join(lines) + "\n"


def finish_reference_set(
    output_dir: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    if len(records) != TARGET_COUNT:
        raise RuntimeError(
            f"Refusing to complete a {len(records)}-track reference set"
        )
    manifest_path = output_dir / "manifest.jsonl"
    checksum_path = output_dir / "manifest.sha256"
    if checksum_path.exists():
        checksum_path.unlink()
    write_jsonl(manifest_path, records)
    attribution_path = output_dir / "ATTRIBUTION.md"
    write_text_atomic(
        attribution_path,
        attribution_text(config["reference_set"], records),
    )
    write_text_atomic(
        checksum_path,
        (
            f"{sha256_file(manifest_path)}  manifest.jsonl\n"
            f"{sha256_file(attribution_path)}  ATTRIBUTION.md\n"
        ),
    )


def dry_run_summary(
    reference_set: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    selected_ids = [record["reference_id"] for record in records[:TARGET_COUNT]]
    print(
        json.dumps(
            {
                "reference_set": reference_set,
                "target_count": TARGET_COUNT,
                "resolved_count": len(selected_ids),
                "selection_sha256": sha256_json(selected_ids),
                "output_directory": str(output_dir),
                "config": config,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    reference_sets = [
        name for name in REFERENCE_SET_NAMES if name in set(args.reference_sets)
    ]
    datasets = import_datasets()
    prompt_checksum, _ = load_frozen_synthetic_prompts()

    for reference_set in reference_sets:
        config = build_config(
            reference_set,
            datasets.__version__,
            prompt_checksum if reference_set == "musicgen-large-v1" else None,
        )
        output_dir = args.output_root.expanduser().resolve() / reference_set
        if args.dry_run:
            records = (
                prepare_synthetic(output_dir, datasets, dry_run=True)
                if reference_set == "musicgen-large-v1"
                else prepare_human(output_dir, datasets, dry_run=True)
            )
            dry_run_summary(reference_set, records, config, output_dir)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        lock = acquire_lock(output_dir)
        try:
            prepare_output(output_dir, config)
            complete = validate_complete(output_dir, config)
            if complete is not None:
                print(
                    f"compatible reference set already exists: "
                    f"{output_dir} ({len(complete)} tracks)"
                )
                continue
            records = (
                prepare_synthetic(output_dir, datasets, dry_run=False)
                if reference_set == "musicgen-large-v1"
                else prepare_human(output_dir, datasets, dry_run=False)
            )
            finish_reference_set(output_dir, config, records)
            print(f"reference set complete: {output_dir}")
        finally:
            lock.close()


if __name__ == "__main__":
    main()
