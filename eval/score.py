import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "audiocraft" / "dataset" / "lofi" / "eval"
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.jsonl"
PROMPTS_CHECKSUM_PATH = Path(__file__).resolve().parent / "prompts.sha256"

SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
AUDIO_SUFFIXES = (".flac", ".mp3", ".ogg", ".wav")
DATASET_COHORT = "dataset_eval"
METRIC_ORDER = ("clap", "kld", "fad")

AUDIOCRAFT_COMMIT = "adf0b04a4452f171970028fcf80f101dd5e26e19"
CLAP_REPOSITORY = "lukewys/laion_clap"
CLAP_CHECKPOINT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CHECKPOINT_SHA256 = (
    "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
)
CLAP_SCORING_SEED = 20260715
CLAP_SAMPLE_RATE = 48_000
KLD_SAMPLE_RATE = 32_000
KLD_PRETRAINED_LENGTH = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a completed eval/generate.py run with CLAP text consistency, "
            "PaSST KLD, and VGGish FAD."
        )
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Existing generated run below runs/.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=(
            "Prepared held-out audio directory. Required by KLD and FAD; "
            "defaults to audiocraft/dataset/lofi/eval."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_ORDER,
        default=list(METRIC_ORDER),
        help="Metrics to compute. Defaults to clap kld fad.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, or 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--clap-checkpoint",
        type=Path,
        help=(
            "Local LAION-CLAP music checkpoint. If omitted, the pinned checkpoint "
            "is downloaded through the Hugging Face cache."
        ),
    )
    parser.add_argument(
        "--clap-batch-size",
        type=int,
        default=4,
        help="CLAP inference batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing score result for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the run and references without loading metric models.",
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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


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


def load_frozen_prompts() -> tuple[str, dict[str, dict[str, Any]]]:
    try:
        checksum_parts = PROMPTS_CHECKSUM_PATH.read_text(encoding="utf-8").split()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Prompt checksum file not found: {PROMPTS_CHECKSUM_PATH}"
        ) from error
    if not checksum_parts or len(checksum_parts[0]) != 64:
        raise RuntimeError(f"Invalid prompt checksum file: {PROMPTS_CHECKSUM_PATH}")
    expected_sha256 = checksum_parts[0].lower()
    actual_sha256 = sha256_file(PROMPTS_PATH)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen prompt manifest checksum mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    prompts = {}
    required = {"id", "cohort", "source_id", "paired_id", "prompt"}
    for record in read_jsonl(PROMPTS_PATH):
        missing = required - record.keys()
        if missing:
            raise RuntimeError(
                f"Frozen prompt is missing {sorted(missing)}: {record.get('id')}"
            )
        prompt_id = record["id"]
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in prompts:
            raise RuntimeError(f"Duplicate or invalid frozen prompt id: {prompt_id!r}")
        prompts[prompt_id] = record
    return actual_sha256, prompts


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def acquire_score_lock(run_dir: Path) -> Any:
    lock_path = run_dir / ".score.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(f"Another scorer is already using {run_dir}") from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def validate_run_name(run_name: str) -> Path:
    if not SAFE_RUN_NAME.fullmatch(run_name):
        raise RuntimeError(
            "--run-name must start with an ASCII letter or digit, contain only "
            "letters, digits, '.', '_', or '-', and be at most 128 characters."
        )
    runs_root = RUNS_ROOT.resolve()
    run_dir = (RUNS_ROOT / run_name).resolve()
    if run_dir.parent != runs_root:
        raise RuntimeError(f"--run-name escapes the runs directory: {run_name!r}")
    if not run_dir.is_dir():
        raise RuntimeError(f"Generated run not found: {run_dir}")
    return run_dir


def resolve_run_audio(run_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError("Run manifest contains an invalid audio_path")
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise RuntimeError(f"Run audio_path must be relative: {relative_path}")
    candidate = run_dir / supplied
    if candidate.is_symlink():
        raise RuntimeError(f"Run audio cannot be a symlink: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as error:
        raise RuntimeError(f"Run audio_path escapes its run: {relative_path}") from error
    if not resolved.is_file():
        raise RuntimeError(f"Run audio is missing or not a regular file: {resolved}")
    return resolved


def load_run(
    run_name: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    run_dir = validate_run_name(run_name)
    config = read_json(run_dir / "config.json")
    if config.get("run_name") != run_name:
        raise RuntimeError(f"Run name does not match {run_dir / 'config.json'}")
    if config.get("schema_version") != 2:
        raise RuntimeError(
            f"Unsupported generated run schema: {config.get('schema_version')!r}"
        )

    prompt_ids = config.get("prompt_ids")
    seeds = config.get("seeds")
    model_source = config.get("model_source")
    audiocraft_commit = config.get("audiocraft_commit")
    generation = config.get("generation")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or not all(isinstance(value, str) and value for value in prompt_ids)
        or len(set(prompt_ids)) != len(prompt_ids)
    ):
        raise RuntimeError("Run config contains invalid prompt_ids")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(value, int) and value >= 0 for value in seeds)
    ):
        raise RuntimeError("Run config contains invalid seeds")
    if not isinstance(model_source, dict) or not model_source:
        raise RuntimeError("Run config contains an invalid model_source")
    if not isinstance(audiocraft_commit, str) or not audiocraft_commit:
        raise RuntimeError("Run config contains an invalid audiocraft_commit")
    if audiocraft_commit != AUDIOCRAFT_COMMIT:
        raise RuntimeError(
            f"Run uses AudioCraft {audiocraft_commit}, but this scorer is pinned to "
            f"{AUDIOCRAFT_COMMIT}"
        )
    if not isinstance(generation, dict):
        raise RuntimeError("Run config contains invalid generation settings")
    configured_duration = generation.get("duration")
    if (
        isinstance(configured_duration, bool)
        or not isinstance(configured_duration, (int, float))
        or not math.isfinite(configured_duration)
        or configured_duration <= 0
    ):
        raise RuntimeError("Run config contains an invalid generation duration")

    prompt_sha256, frozen_prompts = load_frozen_prompts()
    if config.get("prompt_manifest_sha256") != prompt_sha256:
        raise RuntimeError("Run config does not identify the frozen prompt manifest")
    unknown_prompt_ids = sorted(set(prompt_ids) - set(frozen_prompts))
    if unknown_prompt_ids:
        raise RuntimeError(f"Run config contains unknown prompt IDs: {unknown_prompt_ids}")

    expected_clip_ids = {
        f"{prompt_id}__seed-{seed}" for prompt_id in prompt_ids for seed in seeds
    }
    records = read_jsonl(run_dir / "manifest.jsonl")
    by_clip_id = {}
    required = {
        "clip_id",
        "prompt_id",
        "cohort",
        "source_id",
        "paired_id",
        "prompt",
        "seed",
        "audio_path",
        "duration_seconds",
        "sample_rate",
        "model_source",
        "audiocraft_commit",
    }
    for record in records:
        missing = required - record.keys()
        if missing:
            raise RuntimeError(
                f"Run manifest record is missing {sorted(missing)}: {record.get('clip_id')}"
            )
        clip_id = record["clip_id"]
        if not isinstance(clip_id, str) or clip_id in by_clip_id:
            raise RuntimeError(f"Duplicate or invalid clip_id in run manifest: {clip_id!r}")
        if record["prompt_id"] not in prompt_ids or record["seed"] not in seeds:
            raise RuntimeError(f"Unexpected prompt or seed for {clip_id}")
        expected_clip_id = f"{record['prompt_id']}__seed-{record['seed']}"
        if clip_id != expected_clip_id:
            raise RuntimeError(
                f"clip_id does not match its prompt and seed: {clip_id!r}"
            )
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise RuntimeError(f"Invalid prompt for {clip_id}")
        if not isinstance(record["cohort"], str) or not record["cohort"]:
            raise RuntimeError(f"Invalid cohort for {clip_id}")
        frozen_prompt = frozen_prompts[record["prompt_id"]]
        for field in ("cohort", "source_id", "paired_id", "prompt"):
            if record[field] != frozen_prompt[field]:
                raise RuntimeError(
                    f"Frozen prompt field {field} does not match for {clip_id}"
                )
        duration = record["duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
            or duration != configured_duration
        ):
            raise RuntimeError(f"Invalid or inconsistent duration for {clip_id}")
        if (
            isinstance(record["sample_rate"], bool)
            or not isinstance(record["sample_rate"], int)
            or record["sample_rate"] <= 0
        ):
            raise RuntimeError(f"Invalid sample rate for {clip_id}")
        expected_audio_path = f"audio/{clip_id}.wav"
        if record["audio_path"] != expected_audio_path:
            raise RuntimeError(f"Unexpected audio path for {clip_id}")
        if record["model_source"] != model_source:
            raise RuntimeError(f"Model source mismatch for {clip_id}")
        if record["audiocraft_commit"] != audiocraft_commit:
            raise RuntimeError(f"AudioCraft commit mismatch for {clip_id}")
        resolve_run_audio(run_dir, record["audio_path"])
        by_clip_id[clip_id] = record

    actual_clip_ids = set(by_clip_id)
    if actual_clip_ids != expected_clip_ids:
        missing = sorted(expected_clip_ids - actual_clip_ids)
        unexpected = sorted(actual_clip_ids - expected_clip_ids)
        raise RuntimeError(
            "Run is incomplete or inconsistent: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return run_dir, config, records


def resolve_reference_audio(metadata_path: Path) -> Path:
    candidates = [
        metadata_path.with_suffix(suffix)
        for suffix in AUDIO_SUFFIXES
        if metadata_path.with_suffix(suffix).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one audio file beside {metadata_path}, found {candidates}"
        )
    candidate = candidates[0]
    if candidate.is_symlink():
        raise RuntimeError(f"Reference audio cannot be a symlink: {candidate}")
    audio_path = candidate.resolve()
    return audio_path


def load_references(
    reference_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    reference_dir = reference_dir.expanduser().resolve()
    if not reference_dir.is_dir():
        raise RuntimeError(
            f"Held-out reference directory not found: {reference_dir}. "
            "Run prepare.py with enough batches to include the frozen dataset_eval IDs."
        )

    expected_prompts = {}
    for record in records:
        if record["cohort"] != DATASET_COHORT:
            continue
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError(f"dataset_eval clip has no source_id: {record['clip_id']}")
        previous = expected_prompts.setdefault(source_id, record["prompt"])
        if previous != record["prompt"]:
            raise RuntimeError(f"Conflicting prompts for reference source {source_id}")
    if not expected_prompts:
        raise RuntimeError(
            f"Metrics requiring references need at least one {DATASET_COHORT} clip"
        )
    _, frozen_prompts = load_frozen_prompts()
    frozen_reference_ids = {
        prompt["source_id"]
        for prompt in frozen_prompts.values()
        if prompt["cohort"] == DATASET_COHORT
    }
    if set(expected_prompts) != frozen_reference_ids:
        missing = sorted(frozen_reference_ids - set(expected_prompts))
        unexpected = sorted(set(expected_prompts) - frozen_reference_ids)
        raise RuntimeError(
            f"KLD/FAD require the complete frozen {DATASET_COHORT} cohort: "
            f"missing={missing}, unexpected={unexpected}"
        )

    references = {}
    for metadata_path in sorted(reference_dir.glob("*.json")):
        metadata_record = read_json(metadata_path)
        source_id = metadata_record.get("name")
        if source_id not in expected_prompts:
            continue
        if source_id in references:
            raise RuntimeError(f"Duplicate reference metadata for source {source_id}")
        description = metadata_record.get("description")
        if description != expected_prompts[source_id]:
            raise RuntimeError(
                f"Reference prompt mismatch for {source_id}: "
                f"{description!r} != {expected_prompts[source_id]!r}"
            )
        audio_path = resolve_reference_audio(metadata_path)
        references[source_id] = {
            "source_id": source_id,
            "audio_path": audio_path,
            "display_path": display_path(audio_path),
            "sha256": sha256_file(audio_path),
        }

    missing = sorted(set(expected_prompts) - set(references))
    if missing:
        raise RuntimeError(
            "Prepared eval data is missing frozen references for source IDs: "
            + ", ".join(missing)
        )
    return references


def package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


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
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_cuda(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_audio(path: Path) -> tuple[Any, int]:
    try:
        from audiocraft.data.audio import audio_read
    except ImportError as error:
        raise RuntimeError(
            "Scoring requires the AudioCraft environment from this repository's "
            "Docker image."
        ) from error

    waveform, sample_rate = audio_read(path)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise RuntimeError(f"Invalid audio shape for {path}: {tuple(waveform.shape)}")
    if not waveform.isfinite().all():
        raise RuntimeError(f"Audio contains non-finite samples: {path}")
    return waveform, sample_rate


def fixed_duration_audio(
    path: Path,
    duration_seconds: float,
    target_sample_rate: int,
) -> Any:
    import torch
    from audiocraft.data.audio_utils import convert_audio

    waveform, sample_rate = load_audio(path)
    waveform = convert_audio(
        waveform.unsqueeze(0),
        from_rate=sample_rate,
        to_rate=target_sample_rate,
        to_channels=1,
    ).squeeze(0)
    expected_frames = round(duration_seconds * target_sample_rate)
    tolerance_frames = round(0.1 * target_sample_rate)
    if waveform.shape[-1] < expected_frames - tolerance_frames:
        raise RuntimeError(
            f"Audio is shorter than its manifest duration by more than 0.1s: {path}"
        )
    waveform = waveform[..., :expected_frames]
    if waveform.shape[-1] < expected_frames:
        waveform = torch.nn.functional.pad(
            waveform, (0, expected_frames - waveform.shape[-1])
        )
    return waveform


def resolve_clap_checkpoint(supplied: Path | None) -> Path:
    if supplied is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required to download CLAP") from error
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
            f"got {actual_sha256}"
        )
    return checkpoint


def score_clap(
    run_dir: Path,
    records: list[dict[str, Any]],
    device: str,
    batch_size: int,
    checkpoint: Path,
) -> dict[str, float]:
    if batch_size <= 0:
        raise RuntimeError("--clap-batch-size must be greater than zero")
    try:
        import laion_clap
        import torch
    except ImportError as error:
        raise RuntimeError(
            "CLAP scoring dependencies are missing; rebuild the repository Docker image."
        ) from error

    print(f"loading CLAP music checkpoint on {device}...")
    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        amodel="HTSAT-base",
        device=device,
    )
    model.load_ckpt(str(checkpoint), verbose=False)
    model.eval()
    seed_everything(torch, CLAP_SCORING_SEED)

    scores = {}
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            print(
                f"CLAP clips {start + 1}-{start + len(batch)} of {len(records)}"
            )
            audio = torch.stack(
                [
                    fixed_duration_audio(
                        resolve_run_audio(run_dir, record["audio_path"]),
                        float(record["duration_seconds"]),
                        CLAP_SAMPLE_RATE,
                    )
                    for record in batch
                ]
            ).squeeze(1)
            texts = [record["prompt"] for record in batch]
            with torch.inference_mode():
                audio_embeddings = model.get_audio_embedding_from_data(
                    audio, use_tensor=True
                )
                text_embeddings = model.get_text_embedding(texts, use_tensor=True)
                cosine = torch.nn.functional.cosine_similarity(
                    audio_embeddings,
                    text_embeddings,
                    dim=1,
                    eps=1e-8,
                )
            for record, value in zip(batch, cosine.detach().cpu().tolist()):
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"CLAP produced a non-finite score for {record['clip_id']}"
                    )
                scores[record["clip_id"]] = float(value)
    finally:
        del model
        release_cuda(torch)
    return scores


def paired_audio(
    generated_path: Path,
    reference_path: Path,
    duration_seconds: float,
) -> tuple[Any, Any, Any, Any]:
    import torch
    from audiocraft.data.audio_utils import convert_audio

    generated, generated_rate = load_audio(generated_path)
    reference, reference_rate = load_audio(reference_path)
    generated = convert_audio(
        generated.unsqueeze(0),
        from_rate=generated_rate,
        to_rate=KLD_SAMPLE_RATE,
        to_channels=1,
    ).squeeze(0)
    reference = convert_audio(
        reference.unsqueeze(0),
        from_rate=reference_rate,
        to_rate=KLD_SAMPLE_RATE,
        to_channels=1,
    ).squeeze(0)

    maximum_frames = round(duration_seconds * KLD_SAMPLE_RATE)
    frames = min(generated.shape[-1], reference.shape[-1], maximum_frames)
    if frames <= round(0.15 * KLD_SAMPLE_RATE):
        raise RuntimeError(
            f"Generated/reference pair is too short: {generated_path}, {reference_path}"
        )
    generated = generated[..., :frames].unsqueeze(0)
    reference = reference[..., :frames].unsqueeze(0)
    sizes = torch.tensor([frames])
    sample_rates = torch.tensor([KLD_SAMPLE_RATE])
    return generated, reference, sizes, sample_rates


def score_kld(
    run_dir: Path,
    records: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    device: str,
) -> dict[str, dict[str, float | int]]:
    try:
        import torch
        from audiocraft.metrics.kld import PasstKLDivergenceMetric
    except ImportError as error:
        raise RuntimeError(
            "KLD scoring dependencies are missing; rebuild the repository Docker image."
        ) from error

    selected = [record for record in records if record["cohort"] == DATASET_COHORT]
    print(f"loading PaSST KLD model on {device}...")
    metric = PasstKLDivergenceMetric(
        pretrained_length=KLD_PRETRAINED_LENGTH
    ).to(device)
    metric.eval()
    scores = {}
    try:
        for index, record in enumerate(selected, start=1):
            print(f"KLD clip {index} of {len(selected)}: {record['clip_id']}")
            reference = references[record["source_id"]]
            generated_audio, reference_audio, sizes, sample_rates = paired_audio(
                resolve_run_audio(run_dir, record["audio_path"]),
                reference["audio_path"],
                float(record["duration_seconds"]),
            )
            before_pq = float(metric.kld_pq_sum.item())
            before_qp = float(metric.kld_qp_sum.item())
            before_weight = int(metric.weight.item())
            with torch.inference_mode():
                metric.update(
                    generated_audio,
                    reference_audio,
                    sizes,
                    sample_rates,
                )
            segment_count = int(metric.weight.item()) - before_weight
            if segment_count <= 0:
                raise RuntimeError(f"PaSST produced no segments for {record['clip_id']}")
            kld_pq = (float(metric.kld_pq_sum.item()) - before_pq) / segment_count
            kld_qp = (float(metric.kld_qp_sum.item()) - before_qp) / segment_count
            scores[record["clip_id"]] = {
                "kld": kld_pq,
                "kld_pq": kld_pq,
                "kld_qp": kld_qp,
                "kld_both": kld_pq + kld_qp,
                "segment_count": segment_count,
            }
    finally:
        del metric
        release_cuda(torch)
    return scores


def vggish_embeddings(paths: list[Path], device: str) -> Any:
    try:
        import numpy as np
        import torch
        from torchvggish import vggish, vggish_input
    except ImportError as error:
        raise RuntimeError(
            "FAD scoring dependencies are missing; rebuild the repository Docker image."
        ) from error

    model = vggish(postprocess=False)
    # Match the common VGGish FAD configuration: pre-PCA embeddings without
    # the final activation.
    model.embeddings = torch.nn.Sequential(*list(model.embeddings.children())[:-1])
    model.to(device)
    model.device = torch.device(device)
    model.eval()
    embeddings = []
    try:
        for index, path in enumerate(paths, start=1):
            print(f"VGGish file {index} of {len(paths)}: {path.name}")
            waveform, sample_rate = load_audio(path)
            waveform = waveform.mean(dim=0).cpu().numpy()
            examples = vggish_input.waveform_to_examples(waveform, sample_rate)
            with torch.inference_mode():
                value = model.forward(examples.to(device))
            value = value.detach().cpu().numpy()
            if value.ndim != 2 or value.shape[0] == 0:
                raise RuntimeError(f"VGGish produced no embeddings for {path}")
            embeddings.append(value)
    finally:
        del model
        release_cuda(torch)
    return np.concatenate(embeddings, axis=0)


def frechet_distance(first: Any, second: Any) -> float:
    import numpy as np
    from scipy import linalg

    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise RuntimeError(
            f"Incompatible embedding shapes for FAD: {first.shape}, {second.shape}"
        )
    if first.shape[0] < 2 or second.shape[0] < 2:
        raise RuntimeError("FAD requires at least two embeddings in each corpus")

    mu_first = np.mean(first, axis=0)
    mu_second = np.mean(second, axis=0)
    sigma_first = np.cov(first, rowvar=False)
    sigma_second = np.cov(second, rowvar=False)
    difference = mu_first - mu_second
    covariance_mean, _ = linalg.sqrtm(
        sigma_first.dot(sigma_second).astype(complex),
        disp=False,
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma_first.shape[0]) * 1e-6
        covariance_mean = linalg.sqrtm(
            (sigma_first + offset).dot(sigma_second + offset).astype(complex)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(
            np.diagonal(covariance_mean).imag,
            0,
            atol=1e-3,
        ):
            raise RuntimeError(
                "FAD covariance square root has a significant imaginary component"
            )
        covariance_mean = covariance_mean.real

    score = float(
        difference.dot(difference)
        + np.trace(sigma_first)
        + np.trace(sigma_second)
        - 2 * np.trace(covariance_mean)
    )
    if not math.isfinite(score):
        raise RuntimeError("FAD produced a non-finite score")
    if score < -1e-6:
        raise RuntimeError(f"FAD produced a negative score: {score}")
    return max(0.0, score)


def score_fad(
    run_dir: Path,
    records: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    selected = [record for record in records if record["cohort"] == DATASET_COHORT]
    generated_paths = [
        resolve_run_audio(run_dir, record["audio_path"]) for record in selected
    ]
    source_ids = list(dict.fromkeys(record["source_id"] for record in selected))
    reference_paths = [references[source_id]["audio_path"] for source_id in source_ids]

    print(f"extracting VGGish embeddings for {len(generated_paths)} generated clips...")
    generated_embeddings = vggish_embeddings(generated_paths, device)
    print(f"extracting VGGish embeddings for {len(reference_paths)} references...")
    reference_embeddings = vggish_embeddings(reference_paths, device)
    return {
        "value": frechet_distance(reference_embeddings, generated_embeddings),
        "generated_clip_count": len(generated_paths),
        "reference_clip_count": len(reference_paths),
        "generated_embedding_count": int(generated_embeddings.shape[0]),
        "reference_embedding_count": int(reference_embeddings.shape[0]),
    }


def summary(values: list[float], weights: list[int] | None = None) -> dict[str, Any]:
    if not values:
        raise RuntimeError("Cannot summarize an empty metric")
    if weights is None:
        weights = [1] * len(values)
    if len(weights) != len(values) or any(weight <= 0 for weight in weights):
        raise RuntimeError("Invalid metric summary weights")
    total_weight = sum(weights)
    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    variance = (
        sum(
            weight * (value - mean) ** 2
            for value, weight in zip(values, weights)
        )
        / total_weight
    )
    return {
        "count": len(values),
        "weight": total_weight,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def summarize_clap(
    records: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    cohorts = sorted({record["cohort"] for record in records})
    return {
        "direction": "higher_is_better",
        "overall": summary([scores[record["clip_id"]] for record in records]),
        "by_cohort": {
            cohort: summary(
                [
                    scores[record["clip_id"]]
                    for record in records
                    if record["cohort"] == cohort
                ]
            )
            for cohort in cohorts
        },
    }


def summarize_kld(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    selected = [record for record in records if record["cohort"] == DATASET_COHORT]
    metric_names = ("kld", "kld_pq", "kld_qp", "kld_both")
    return {
        "direction": "lower_is_better",
        "cohort": DATASET_COHORT,
        "scores": {
            metric_name: summary(
                [float(scores[record["clip_id"]][metric_name]) for record in selected],
                [
                    int(scores[record["clip_id"]]["segment_count"])
                    for record in selected
                ],
            )
            for metric_name in metric_names
        },
    }


def make_clip_records(
    run_dir: Path,
    records: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    clap_scores: dict[str, float],
    kld_scores: dict[str, dict[str, float | int]],
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        audio_path = resolve_run_audio(run_dir, record["audio_path"])
        clip_metrics: dict[str, Any] = {}
        if record["clip_id"] in clap_scores:
            clip_metrics["clap"] = clap_scores[record["clip_id"]]
        if record["clip_id"] in kld_scores:
            clip_metrics.update(kld_scores[record["clip_id"]])

        output_record = {
            "clip_id": record["clip_id"],
            "prompt_id": record["prompt_id"],
            "cohort": record["cohort"],
            "source_id": record.get("source_id"),
            "seed": record["seed"],
            "prompt": record["prompt"],
            "audio_path": record["audio_path"],
            "audio_sha256": sha256_file(audio_path),
            "metrics": clip_metrics,
        }
        source_id = record.get("source_id")
        if source_id in references:
            reference = references[source_id]
            output_record["reference"] = {
                "audio_path": reference["display_path"],
                "audio_sha256": reference["sha256"],
            }
        output.append(output_record)
    return output


def make_score_config(
    args: argparse.Namespace,
    run_dir: Path,
    metrics: list[str],
    references: dict[str, dict[str, Any]],
    clip_records: list[dict[str, Any]],
    clap_checkpoint: Path | None,
    device: str,
) -> dict[str, Any]:
    reference_fingerprints = {
        source_id: reference["sha256"]
        for source_id, reference in sorted(references.items())
    }
    audio_fingerprints = {
        record["clip_id"]: record["audio_sha256"] for record in clip_records
    }
    return {
        "schema_version": 1,
        "run_name": args.run_name,
        "run_config_sha256": sha256_file(run_dir / "config.json"),
        "run_manifest_sha256": sha256_file(run_dir / "manifest.jsonl"),
        "generated_audio_sha256": sha256_json(audio_fingerprints),
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
        "metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            "device": device,
            "torch": package_version("torch"),
            "torchaudio": package_version("torchaudio"),
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
            "resampy": package_version("resampy"),
            "scipy": package_version("scipy"),
            "timm": package_version("timm"),
            "transformers": package_version("transformers"),
            "audiocraft": package_version("audiocraft"),
            "laion-clap": (
                package_version("laion-clap") if "clap" in metrics else None
            ),
            "hear21passt": (
                package_version("hear21passt") if "kld" in metrics else None
            ),
            "torchvggish": (
                package_version("torchvggish") if "fad" in metrics else None
            ),
        },
        "reference_set": (
            {
                "directory": display_path(args.reference_dir),
                "cohort": DATASET_COHORT,
                "source_count": len(references),
                "sha256": sha256_json(reference_fingerprints),
            }
            if references
            else None
        ),
        "clap": (
            {
                "repository": CLAP_REPOSITORY,
                "checkpoint": CLAP_CHECKPOINT_NAME,
                "checkpoint_sha256": CLAP_CHECKPOINT_SHA256,
                "architecture": "HTSAT-base",
                "enable_fusion": False,
                "sample_rate": CLAP_SAMPLE_RATE,
                "scoring_seed": CLAP_SCORING_SEED,
                "batch_size": args.clap_batch_size,
                "resolved_checkpoint": display_path(clap_checkpoint),
            }
            if clap_checkpoint is not None
            else None
        ),
        "kld": (
            {
                "classifier": "PaSST",
                "pretrained_length_seconds": KLD_PRETRAINED_LENGTH,
                "sample_rate": KLD_SAMPLE_RATE,
            }
            if "kld" in metrics
            else None
        ),
        "fad": (
            {
                "embedding_model": "VGGish",
                "implementation": "torchvggish_pre_pca_without_final_activation",
                "cohort": DATASET_COHORT,
            }
            if "fad" in metrics
            else None
        ),
    }


def existing_result_action(
    run_dir: Path,
    score_config: dict[str, Any],
    overwrite: bool,
) -> bool:
    paths = [
        run_dir / "score_config.json",
        run_dir / "clip_metrics.jsonl",
        run_dir / "metrics.json",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return False
    if overwrite:
        return False
    if len(existing) == len(paths):
        existing_config = read_json(paths[0])
        output_hashes = existing_config.pop("outputs", None)
        valid_outputs = (
            isinstance(output_hashes, dict)
            and output_hashes.get("clip_metrics_sha256") == sha256_file(paths[1])
            and output_hashes.get("metrics_sha256") == sha256_file(paths[2])
        )
        if valid_outputs and existing_config == score_config:
            print(f"compatible score result already exists: {run_dir}")
            return True
    raise RuntimeError(
        f"Score output already exists or is incomplete in {run_dir}; "
        "use --overwrite to replace it."
    )


def main() -> None:
    args = parse_args()
    metrics = [name for name in METRIC_ORDER if name in set(args.metrics)]
    run_dir, run_config, records = load_run(args.run_name)
    needs_references = bool({"kld", "fad"} & set(metrics))
    references = (
        load_references(args.reference_dir, records) if needs_references else {}
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "metrics": metrics,
                    "clip_count": len(records),
                    "cohorts": {
                        cohort: sum(
                            record["cohort"] == cohort for record in records
                        )
                        for cohort in sorted(
                            {record["cohort"] for record in records}
                        )
                    },
                    "reference_count": len(references),
                    "clap_checkpoint": (
                        str(args.clap_checkpoint.expanduser())
                        if args.clap_checkpoint is not None
                        else f"hf://{CLAP_REPOSITORY}/{CLAP_CHECKPOINT_NAME}"
                    ),
                    "run_model_source": run_config.get("model_source"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Scoring requires the repository's AudioCraft Docker environment."
        ) from error
    score_lock = acquire_score_lock(run_dir)
    device = select_device(args.device, torch)
    clap_checkpoint = (
        resolve_clap_checkpoint(args.clap_checkpoint) if "clap" in metrics else None
    )

    initial_clip_records = make_clip_records(
        run_dir,
        records,
        references,
        {},
        {},
    )
    score_config = make_score_config(
        args,
        run_dir,
        metrics,
        references,
        initial_clip_records,
        clap_checkpoint,
        device,
    )
    if existing_result_action(run_dir, score_config, args.overwrite):
        return

    clap_scores: dict[str, float] = {}
    kld_scores: dict[str, dict[str, float | int]] = {}
    fad_result: dict[str, Any] | None = None

    if "clap" in metrics:
        assert clap_checkpoint is not None
        clap_scores = score_clap(
            run_dir,
            records,
            device,
            args.clap_batch_size,
            clap_checkpoint,
        )
    if "kld" in metrics:
        kld_scores = score_kld(run_dir, records, references, device)
    if "fad" in metrics:
        fad_result = score_fad(run_dir, records, references, device)

    clip_records = initial_clip_records
    for record in clip_records:
        clip_id = record["clip_id"]
        if clip_id in clap_scores:
            record["metrics"]["clap"] = clap_scores[clip_id]
        if clip_id in kld_scores:
            record["metrics"].update(kld_scores[clip_id])

    metric_results: dict[str, Any] = {}
    if clap_scores:
        metric_results["clap"] = {
            **summarize_clap(records, clap_scores),
            "implementation": {
                "package": "laion-clap",
                "package_version": package_version("laion-clap"),
                "checkpoint": CLAP_CHECKPOINT_NAME,
                "checkpoint_sha256": CLAP_CHECKPOINT_SHA256,
                "architecture": "HTSAT-base",
            },
        }
    if kld_scores:
        metric_results["kld"] = {
            **summarize_kld(records, kld_scores),
            "implementation": {
                "package": "hear21passt",
                "package_version": package_version("hear21passt"),
                "audiocraft_version": package_version("audiocraft"),
                "classifier": "PaSST",
            },
        }
    if fad_result is not None:
        metric_results["fad"] = {
            "direction": "lower_is_better",
            "cohort": DATASET_COHORT,
            **fad_result,
            "implementation": {
                "package": "torchvggish",
                "package_version": package_version("torchvggish"),
                "embedding": "VGGish pre-PCA without final activation",
            },
            "interpretation": (
                "The references are MusicGen-Large-generated dataset audio, so this "
                "measures similarity to that synthetic domain rather than general "
                "recorded-music realism. The small frozen reference set also makes "
                "the estimate noisy."
            ),
        }

    metrics_output = {
        "schema_version": 1,
        "run_name": args.run_name,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "torch": torch.__version__,
        "metrics": metric_results,
    }
    completion_path = run_dir / "score_config.json"
    if completion_path.exists():
        completion_path.unlink()
    write_jsonl(run_dir / "clip_metrics.jsonl", clip_records)
    write_json(run_dir / "metrics.json", metrics_output)
    # Write the locked config last so its presence marks a complete score result.
    completion_config = {
        **score_config,
        "outputs": {
            "clip_metrics_sha256": sha256_file(run_dir / "clip_metrics.jsonl"),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        },
    }
    write_json(completion_path, completion_config)
    print(f"score complete: {run_dir / 'metrics.json'}")
    score_lock.close()


if __name__ == "__main__":
    main()
