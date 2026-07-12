import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import datasets
from keybert import KeyBERT
from torch.utils.data import DataLoader


SAMPLE_RATE = 44_100
TRAIN_SIZE = 0.8
SHUFFLE_SEED = 42
BATCH_SIZE = 32
REPO = "vikhyatk/lofi"

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIOCRAFT_ROOT = PROJECT_ROOT / "audiocraft"
DATASET_ROOT = AUDIOCRAFT_ROOT / "dataset" / "lofi"
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


def check_audiocraft_checkout() -> None:
    audio_dataset_module = AUDIOCRAFT_ROOT / "audiocraft" / "data" / "audio_dataset.py"
    if not audio_dataset_module.is_file():
        raise RuntimeError(
            "The AudioCraft submodule is not initialized. Run "
            "`git submodule update --init --recursive` first."
        )


def write_dataset_config() -> None:
    config_path = AUDIOCRAFT_ROOT / "config" / "dset" / "audio" / "lofi.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""# @package __global__

datasource:
    max_sample_rate: {SAMPLE_RATE}
    max_channels: 1

    train: egs/train
    valid: egs/eval
    evaluate: egs/eval
    generate: egs/train
""",
        encoding="utf-8",
    )


def reset_split_directories() -> None:
    for split in SPLITS:
        split_path = DATASET_ROOT / split
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True)


def split_for_track(track_id: str) -> str:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    split_value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if split_value < TRAIN_SIZE else "eval"


def filename_for_track(track_id: str) -> str:
    readable_id = re.sub(r"[^A-Za-z0-9._-]+", "-", track_id).strip(".-")
    readable_id = readable_id[:80] or "track"
    digest = hashlib.sha256(track_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable_id}-{digest}"


def music_metadata(
    track_id: str,
    prompt: str,
    extracted_keywords: list[tuple[str, float]],
) -> dict:
    prompt_lower = prompt.lower()
    instrument = next((name for name in INSTRUMENTS if name in prompt_lower), None)
    moods = [mood for mood in MOODS if mood in prompt_lower]

    # MusicDataset requires these keys when info_fields_required is enabled.
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


def write_track(
    track_id: str,
    prompt: str,
    audio_bytes: bytes,
    extracted_keywords: list[tuple[str, float]],
) -> str:
    split = split_for_track(track_id)
    track_stem = DATASET_ROOT / split / filename_for_track(track_id)
    Path(f"{track_stem}.mp3").write_bytes(audio_bytes)
    Path(f"{track_stem}.json").write_text(
        json.dumps(
            music_metadata(track_id, prompt, extracted_keywords),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return split


def generate_manifest(split: str, expected_tracks: int) -> None:
    manifest_path = AUDIOCRAFT_ROOT / "egs" / split / "data.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "audiocraft.data.audio_dataset",
            str((DATASET_ROOT / split).relative_to(AUDIOCRAFT_ROOT)),
            str(manifest_path.relative_to(AUDIOCRAFT_ROOT)),
        ],
        cwd=AUDIOCRAFT_ROOT,
        check=True,
    )
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest_tracks = sum(1 for line in manifest_file if line.strip())
    if manifest_tracks != expected_tracks:
        raise RuntimeError(
            f"AudioCraft added {manifest_tracks} of {expected_tracks} {split} tracks "
            "to the manifest. Check its preceding audio errors."
        )


def read_batch_count() -> int:
    try:
        batch_count = int(os.environ["BATCH_COUNT"])
    except KeyError as error:
        raise RuntimeError("Set BATCH_COUNT to the number of batches to prepare.") from error
    except ValueError as error:
        raise RuntimeError("BATCH_COUNT must be an integer.") from error

    if batch_count <= 0:
        raise RuntimeError("BATCH_COUNT must be greater than zero.")
    return batch_count


def main() -> None:
    batch_count = read_batch_count()
    check_audiocraft_checkout()
    write_dataset_config()
    reset_split_directories()

    dataset = datasets.load_dataset(REPO, split="train", streaming=True)
    dataset = dataset.shuffle(seed=SHUFFLE_SEED)
    dataset = dataset.cast_column(
        "audio",
        datasets.Audio(sampling_rate=SAMPLE_RATE, decode=False),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    keyword_model = KeyBERT()
    split_counts = {split: 0 for split in SPLITS}
    seen_track_ids = set()

    for batch_index, row in enumerate(loader):
        if batch_index >= batch_count:
            break
        print(f"processing batch {batch_index + 1}/{batch_count}...")

        batch_keywords = keyword_model.extract_keywords(row["prompt"])
        for index, audio_bytes in enumerate(row["audio"]["bytes"]):
            track_id = str(row["id"][index])
            prompt = row["prompt"][index].strip()
            if any(word in prompt.lower() for word in IGNORE_WORDS):
                continue
            if track_id in seen_track_ids:
                raise RuntimeError(f"Duplicate track id encountered: {track_id}")
            seen_track_ids.add(track_id)

            split = write_track(
                track_id,
                prompt,
                audio_bytes,
                batch_keywords[index],
            )
            split_counts[split] += 1

    for split, count in split_counts.items():
        if count == 0:
            raise RuntimeError(f"The deterministic split produced no {split} tracks.")
        generate_manifest(split, count)

    print(
        "prepared "
        + ", ".join(f"{count} {split} tracks" for split, count in split_counts.items())
    )


if __name__ == "__main__":
    main()
