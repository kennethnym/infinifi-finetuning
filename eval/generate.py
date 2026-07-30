import argparse
from datetime import datetime, timezone
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
EVAL_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = EVAL_DIR / "prompts.jsonl"
CHECKSUM_PATH = EVAL_DIR / "prompts.sha256"
RUNS_ROOT = PROJECT_ROOT / "runs"

AUDIOCRAFT_COMMIT = "adf0b04a4452f171970028fcf80f101dd5e26e19"
AUDIOCRAFT_PATCH = PROJECT_ROOT / "patches" / "audiocraft-lora.patch"
BACKENDS = ("musicgen", "ace-step")
DEFAULT_BACKEND = "musicgen"
ACE_STEP_DEFAULT_MODEL = "ACE-Step/acestep-v15-xl-turbo-diffusers"
ACE_STEP_DEFAULT_REVISION = "200ba991ae448051e14b0183157e35c2d27c9fb0"
ACE_STEP_DIFFUSERS_VERSION = "0.39.0"
ACE_STEP_LYRICS = "[Instrumental]"
ACE_STEP_DEFAULT_PARAMS = {
    "duration": 30,
    "task_type": "text2music",
    "lyrics": ACE_STEP_LYRICS,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "shift": 3.0,
}
DEFAULT_GENERATION_PARAMS = {
    "duration": 30,
    "use_sampling": True,
    "top_k": 250,
    "top_p": 0.0,
    "temperature": 1.0,
    "cfg_coef": 3.0,
}
AUDIO_WRITE_PARAMS = {
    "format": "wav",
    "strategy": "loudness",
    "loudness_compressor": True,
}
ACE_STEP_AUDIO_WRITE_PARAMS = {
    **AUDIO_WRITE_PARAMS,
    "subtype": "PCM_16",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LOCAL_MODEL_FILES = ("state_dict.bin", "compression_state_dict.bin")
ADAPTER_METADATA = "adapter.json"
ADAPTER_WEIGHTS = "adapter_state.bin"
ADAPTER_FILES = (ADAPTER_METADATA, ADAPTER_WEIGHTS)
DEFAULT_ADAPTER_SCALE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reproducible evaluation run with MusicGen or ACE-Step 1.5."
        )
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help="Generation backend (default: musicgen).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Pretrained model ID. MusicGen also accepts an exported local model "
            "or LoRA adapter package."
        ),
    )
    parser.add_argument(
        "--model-revision",
        help=(
            "Exact Hugging Face revision for ACE-Step. The documented XL Turbo "
            "model defaults to the revision pinned by this repository."
        ),
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Safe identifier used as the directory name below runs/.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45],
        help="One or more fixed generation seeds.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, or 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generate up to this many prompts in parallel (default: 1).",
    )
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=DEFAULT_GENERATION_PARAMS["cfg_coef"],
        help="MusicGen classifier-free guidance coefficient (default: 3.0).",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=DEFAULT_ADAPTER_SCALE,
        help=(
            "Inference-only multiplier for LoRA projections "
            "(default: 1.0; LoRA adapters only)."
        ),
    )
    parser.add_argument(
        "--ace-steps",
        type=int,
        default=ACE_STEP_DEFAULT_PARAMS["num_inference_steps"],
        help="ACE-Step denoising steps (default: 8 for XL Turbo).",
    )
    parser.add_argument(
        "--ace-guidance-scale",
        type=float,
        default=ACE_STEP_DEFAULT_PARAMS["guidance_scale"],
        help=(
            "ACE-Step guidance scale (default: 1.0; XL Turbo has distilled "
            "guidance and ignores values above 1.0)."
        ),
    )
    parser.add_argument(
        "--ace-shift",
        type=float,
        default=ACE_STEP_DEFAULT_PARAMS["shift"],
        help="ACE-Step timestep shift (default: 3.0 for XL Turbo).",
    )
    parser.add_argument(
        "--ace-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="ACE-Step model dtype (default: bfloat16 on CUDA, float32 on CPU).",
    )
    parser.add_argument(
        "--ace-cpu-offload",
        action="store_true",
        help="Offload ACE-Step pipeline components to CPU to reduce CUDA memory use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N prompts. Intended for a distinct smoke-test run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the locked run configuration without generating.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def verify_prompt_checksum() -> str:
    try:
        checksum_parts = CHECKSUM_PATH.read_text(encoding="utf-8").split()
    except FileNotFoundError as error:
        raise RuntimeError(f"Prompt checksum file not found: {CHECKSUM_PATH}") from error

    if not checksum_parts or len(checksum_parts[0]) != 64:
        raise RuntimeError(f"Invalid prompt checksum file: {CHECKSUM_PATH}")

    expected = checksum_parts[0].lower()
    actual = sha256_file(PROMPTS_PATH)
    if actual != expected:
        raise RuntimeError(
            f"Prompt manifest checksum mismatch: expected {expected}, got {actual}. "
            "Create a new prompt-set version instead of modifying the frozen manifest."
        )
    return actual


def load_prompts() -> list[dict[str, Any]]:
    prompts = []
    seen_ids = set()
    seen_text = set()

    try:
        lines = PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise RuntimeError(f"Prompt manifest not found: {PROMPTS_PATH}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            prompt = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSON in {PROMPTS_PATH} at line {line_number}: {error}"
            ) from error

        missing = {"id", "cohort", "prompt"} - prompt.keys()
        if missing:
            raise RuntimeError(
                f"Prompt at line {line_number} is missing: {sorted(missing)}"
            )
        if not isinstance(prompt["id"], str) or not SAFE_ID.fullmatch(prompt["id"]):
            raise RuntimeError(f"Unsafe prompt id at line {line_number}: {prompt['id']!r}")
        if not isinstance(prompt["cohort"], str) or not prompt["cohort"]:
            raise RuntimeError(f"Invalid cohort at line {line_number}")
        if not isinstance(prompt["prompt"], str) or not prompt["prompt"].strip():
            raise RuntimeError(f"Empty prompt text at line {line_number}")
        if prompt["id"] in seen_ids:
            raise RuntimeError(f"Duplicate prompt id: {prompt['id']}")
        if prompt["prompt"] in seen_text:
            raise RuntimeError(f"Duplicate prompt text at line {line_number}")

        seen_ids.add(prompt["id"])
        seen_text.add(prompt["prompt"])
        prompts.append(prompt)

    if not prompts:
        raise RuntimeError(f"No prompts found in {PROMPTS_PATH}")
    return prompts


def validate_args(args: argparse.Namespace, prompt_count: int) -> Path:
    backend = getattr(args, "backend", DEFAULT_BACKEND)
    if backend not in BACKENDS:
        raise RuntimeError(f"Unsupported generation backend: {backend!r}")
    if not SAFE_RUN_NAME.fullmatch(args.run_name):
        raise RuntimeError(
            "--run-name must start with an ASCII letter or digit, contain only "
            "letters, digits, '.', '_', or '-', and be at most 128 characters."
        )
    if args.limit is not None and not 1 <= args.limit <= prompt_count:
        raise RuntimeError(f"--limit must be between 1 and {prompt_count}")
    if not args.seeds:
        raise RuntimeError("At least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise RuntimeError("Seeds must be unique")
    if any(seed < 0 for seed in args.seeds):
        raise RuntimeError("Seeds must be non-negative")
    if not args.model.strip():
        raise RuntimeError("--model cannot be empty")
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be greater than zero")
    if not math.isfinite(args.cfg_coef) or args.cfg_coef <= 0:
        raise RuntimeError("--cfg-coef must be a finite positive number")
    if not math.isfinite(args.adapter_scale) or args.adapter_scale < 0:
        raise RuntimeError("--adapter-scale must be a finite non-negative number")
    if backend == "ace-step":
        if getattr(args, "ace_steps", ACE_STEP_DEFAULT_PARAMS["num_inference_steps"]) <= 0:
            raise RuntimeError("--ace-steps must be greater than zero")
        ace_guidance_scale = getattr(
            args,
            "ace_guidance_scale",
            ACE_STEP_DEFAULT_PARAMS["guidance_scale"],
        )
        if not math.isfinite(ace_guidance_scale) or ace_guidance_scale <= 0:
            raise RuntimeError(
                "--ace-guidance-scale must be a finite positive number"
            )
        ace_shift = getattr(args, "ace_shift", ACE_STEP_DEFAULT_PARAMS["shift"])
        if not math.isfinite(ace_shift) or ace_shift <= 0:
            raise RuntimeError("--ace-shift must be a finite positive number")
        if args.cfg_coef != DEFAULT_GENERATION_PARAMS["cfg_coef"]:
            raise RuntimeError("--cfg-coef only applies to the MusicGen backend")
        if args.adapter_scale != DEFAULT_ADAPTER_SCALE:
            raise RuntimeError("--adapter-scale only applies to MusicGen LoRA adapters")
    else:
        if getattr(args, "model_revision", None) is not None:
            raise RuntimeError("--model-revision only applies to the ACE-Step backend")
        if getattr(args, "ace_cpu_offload", False):
            raise RuntimeError("--ace-cpu-offload only applies to the ACE-Step backend")

    runs_root = RUNS_ROOT.resolve()
    output_dir = (RUNS_ROOT / args.run_name).resolve()
    if output_dir.parent != runs_root:
        raise RuntimeError(f"--run-name escapes the runs directory: {args.run_name!r}")
    return output_dir


def generation_params(cfg_coef: float) -> dict[str, Any]:
    return {
        **DEFAULT_GENERATION_PARAMS,
        "cfg_coef": cfg_coef,
    }


def ace_step_generation_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **ACE_STEP_DEFAULT_PARAMS,
        "num_inference_steps": getattr(
            args,
            "ace_steps",
            ACE_STEP_DEFAULT_PARAMS["num_inference_steps"],
        ),
        "guidance_scale": getattr(
            args,
            "ace_guidance_scale",
            ACE_STEP_DEFAULT_PARAMS["guidance_scale"],
        ),
        "shift": getattr(
            args,
            "ace_shift",
            ACE_STEP_DEFAULT_PARAMS["shift"],
        ),
    }


def model_package_digest(directory: Path) -> tuple[str, int]:
    files = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Local model packages cannot contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    files.sort(key=lambda path: path.relative_to(directory).as_posix())
    if not files:
        raise RuntimeError(f"Local model package contains no regular files: {directory}")

    digest = hashlib.sha256()
    digest.update(b"infinifi-model-package-v1\0")
    for path in files:
        relative_path = path.relative_to(directory).as_posix().encode("utf-8")
        file_size = path.stat().st_size
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(file_size.to_bytes(16, "big"))
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def load_adapter_metadata(directory: Path) -> dict[str, Any]:
    missing = [
        filename
        for filename in ADAPTER_FILES
        if not (directory / filename).is_file()
        or (directory / filename).is_symlink()
    ]
    if missing:
        raise RuntimeError(
            f"Local adapter package is incomplete; missing regular files: {missing}"
        )
    metadata_path = directory / ADAPTER_METADATA
    try:
        adapter = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid adapter metadata: {metadata_path}") from error
    fixed = {
        "schema_version": 1,
        "format": "infinifi_musicgen_lora",
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
        "audiocraft_patch_sha256": sha256_file(AUDIOCRAFT_PATCH),
    }
    for key, expected in fixed.items():
        if adapter.get(key) != expected:
            raise RuntimeError(
                f"Adapter metadata {key} does not match: "
                f"expected {expected!r}, got {adapter.get(key)!r}"
            )
    base_model = adapter.get("base_model")
    if not isinstance(base_model, str) or not base_model:
        raise RuntimeError("Adapter metadata has no base model")
    lora = adapter.get("lora")
    if not isinstance(lora, dict) or not lora.get("enabled"):
        raise RuntimeError("Adapter metadata has no enabled LoRA configuration")
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    dropout = lora.get("dropout")
    condition_gated = lora.get("condition_gated", True)
    targets = lora.get("targets")
    supported_targets = {
        "self_attention",
        "cross_attention",
        "feedforward",
    }
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise RuntimeError("Adapter metadata has an invalid LoRA rank")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise RuntimeError("Adapter metadata has an invalid LoRA alpha")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0 <= dropout < 1
    ):
        raise RuntimeError("Adapter metadata has an invalid LoRA dropout")
    if not isinstance(condition_gated, bool):
        raise RuntimeError(
            "Adapter metadata has an invalid LoRA condition_gated value"
        )
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) for target in targets)
        or set(targets) - supported_targets
    ):
        raise RuntimeError("Adapter metadata has invalid LoRA targets")
    recorded_weights = adapter.get("files", {}).get(ADAPTER_WEIGHTS)
    actual_weights = {
        "sha256": sha256_file(directory / ADAPTER_WEIGHTS),
        "size_bytes": (directory / ADAPTER_WEIGHTS).stat().st_size,
    }
    if recorded_weights != actual_weights:
        raise RuntimeError("Adapter weight hash or size does not match adapter metadata")
    return adapter


def resolve_model_source(
    supplied_source: str,
    backend: str = DEFAULT_BACKEND,
    model_revision: str | None = None,
) -> tuple[str, dict[str, Any]]:
    if backend == "ace-step":
        candidate = Path(supplied_source).expanduser()
        if candidate.exists() or candidate.is_absolute() or supplied_source.startswith(
            ("./", "../", "~")
        ):
            raise RuntimeError(
                "ACE-Step currently requires a Hugging Face Diffusers model ID, "
                "not a local model directory."
            )
        revision = model_revision
        if revision is None and supplied_source == ACE_STEP_DEFAULT_MODEL:
            revision = ACE_STEP_DEFAULT_REVISION
        if revision is None:
            raise RuntimeError(
                "--model-revision is required for ACE-Step models other than "
                f"{ACE_STEP_DEFAULT_MODEL}."
            )
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise RuntimeError(
                "--model-revision must be a full 40-character Hugging Face commit."
            )
        return supplied_source, {
            "type": "pretrained",
            "backend": "ace-step",
            "supplied": supplied_source,
            "model_id": supplied_source,
            "revision": revision.lower(),
            "library": "diffusers",
            "library_version": ACE_STEP_DIFFUSERS_VERSION,
        }
    if backend != "musicgen":
        raise RuntimeError(f"Unsupported generation backend: {backend!r}")

    candidate = Path(supplied_source).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise RuntimeError(f"Local model source is not a directory: {candidate}")
        directory = candidate.resolve()
        if (directory / ADAPTER_METADATA).exists() or (directory / ADAPTER_WEIGHTS).exists():
            adapter = load_adapter_metadata(directory)
            package_sha256, file_count = model_package_digest(directory)
            return str(directory), {
                "type": "lora_adapter",
                "backend": "musicgen",
                "supplied": supplied_source,
                "path": display_path(directory),
                "package_sha256": package_sha256,
                "file_count": file_count,
                "base_model": adapter["base_model"],
                "lora": adapter["lora"],
                "adapter_weights_sha256": adapter["files"][ADAPTER_WEIGHTS]["sha256"],
                "audiocraft_commit": AUDIOCRAFT_COMMIT,
                "audiocraft_patch_sha256": adapter["audiocraft_patch_sha256"],
            }
        missing = [
            filename
            for filename in LOCAL_MODEL_FILES
            if not (directory / filename).is_file()
            or (directory / filename).is_symlink()
        ]
        if missing:
            raise RuntimeError(
                f"Local model package is incomplete; missing regular files: {missing}"
            )
        package_sha256, file_count = model_package_digest(directory)
        return str(directory), {
            "type": "local_package",
            "backend": "musicgen",
            "supplied": supplied_source,
            "path": display_path(directory),
            "package_sha256": package_sha256,
            "file_count": file_count,
        }

    if candidate.is_absolute() or supplied_source.startswith(("./", "../", "~")):
        raise RuntimeError(f"Local model package directory not found: {candidate}")

    return supplied_source, {
        "type": "pretrained",
        "backend": "musicgen",
        "supplied": supplied_source,
        "model_id": supplied_source,
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
    }


def load_musicgen_model(
    MusicGen: Any,
    torch: Any,
    model_source: str,
    model_source_record: dict[str, Any],
    device: str,
) -> Any:
    if model_source_record["type"] != "lora_adapter":
        return MusicGen.get_pretrained(model_source, device=device)

    try:
        from audiocraft.modules.lora import inject_lora, load_adapter_state_dict
    except ImportError as error:
        raise RuntimeError(
            "LoRA adapter generation requires the patched AudioCraft build."
        ) from error

    adapter_directory = Path(model_source)
    model = MusicGen.get_pretrained(
        model_source_record["base_model"],
        device=device,
    )
    inject_lora(model.lm, model_source_record["lora"])
    package = torch.load(
        adapter_directory / ADAPTER_WEIGHTS,
        map_location="cpu",
    )
    if (
        not isinstance(package, dict)
        or package.get("format") != "infinifi_musicgen_lora"
        or not isinstance(package.get("state_dict"), dict)
    ):
        raise RuntimeError(
            f"Invalid LoRA adapter weights: {adapter_directory / ADAPTER_WEIGHTS}"
        )
    load_adapter_state_dict(model.lm, package["state_dict"])
    model.lm.eval()
    return model


def apply_adapter_scale(module: Any, scale: float) -> int:
    try:
        from audiocraft.modules.lora import GatedLoRAProjection
    except ImportError as error:
        raise RuntimeError(
            "LoRA adapter scaling requires the patched AudioCraft build."
        ) from error

    projections = [
        child
        for child in module.modules()
        if isinstance(child, GatedLoRAProjection)
    ]
    if not projections:
        raise RuntimeError("Loaded LoRA model contains no adapter projections")
    for projection in projections:
        projection.scaling *= scale
    return len(projections)


def build_locked_config(
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    prompt_sha256: str,
    model_source: dict[str, Any],
) -> dict[str, Any]:
    if (
        model_source.get("type") != "lora_adapter"
        and args.adapter_scale != DEFAULT_ADAPTER_SCALE
    ):
        raise RuntimeError("--adapter-scale only applies to LoRA adapter models")
    backend = model_source.get(
        "backend",
        getattr(args, "backend", DEFAULT_BACKEND),
    )
    if backend not in BACKENDS:
        raise RuntimeError(f"Unsupported generation backend: {backend!r}")
    config = {
        "schema_version": 3,
        "run_name": args.run_name,
        "backend": backend,
        "model_source": model_source,
        "adapter_scale": args.adapter_scale,
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "prompt_manifest": display_path(PROMPTS_PATH),
        "prompt_manifest_sha256": prompt_sha256,
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "generation": (
            generation_params(args.cfg_coef)
            if backend == "musicgen"
            else ace_step_generation_params(args)
        ),
        "audio_write": (
            AUDIO_WRITE_PARAMS
            if backend == "musicgen"
            else ACE_STEP_AUDIO_WRITE_PARAMS
        ),
    }
    if backend == "musicgen":
        config["audiocraft_commit"] = AUDIOCRAFT_COMMIT
    else:
        config["ace_dtype"] = getattr(args, "ace_dtype", "auto")
        config["ace_cpu_offload"] = getattr(args, "ace_cpu_offload", False)
    return config


def clip_backend_metadata(locked_config: dict[str, Any]) -> dict[str, Any]:
    backend = locked_config.get("backend", DEFAULT_BACKEND)
    metadata = {"backend": backend}
    if backend == "musicgen":
        metadata["audiocraft_commit"] = AUDIOCRAFT_COMMIT
    return metadata


def build_clip_plan(
    prompts: list[dict[str, Any]], seeds: list[int]
) -> list[dict[str, Any]]:
    plan = []
    for prompt in prompts:
        for seed in seeds:
            clip_id = f"{prompt['id']}__seed-{seed}"
            plan.append(
                {
                    "clip_id": clip_id,
                    "prompt_id": prompt["id"],
                    "cohort": prompt["cohort"],
                    "source_id": prompt.get("source_id"),
                    "paired_id": prompt.get("paired_id"),
                    "prompt": prompt["prompt"],
                    "seed": seed,
                    "audio_path": f"audio/{clip_id}.wav",
                }
            )
    return plan


def build_generation_batches(
    clip_plan: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    clips_by_seed: dict[int, list[dict[str, Any]]] = {}
    for clip in clip_plan:
        clips_by_seed.setdefault(clip["seed"], []).append(clip)

    batches = []
    for clips in clips_by_seed.values():
        for start in range(0, len(clips), batch_size):
            batches.append(clips[start : start + batch_size])
    return batches


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def prepare_output(
    output_dir: Path, locked_config: dict[str, Any]
) -> tuple[Path, dict[str, dict[str, Any]]]:
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.jsonl"
    audio_dir = output_dir / "audio"

    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {output_dir}")

    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid run configuration: {config_path}") from error
        if existing_config != locked_config:
            raise RuntimeError(
                f"Run configuration differs from {config_path}; use a new run name."
            )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is non-empty but has no config: {output_dir}"
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir()
        write_json(config_path, locked_config)

    if not audio_dir.is_dir():
        raise RuntimeError(f"Run audio directory is missing: {audio_dir}")

    completed = {}
    if manifest_path.exists():
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                clip_id = record["clip_id"]
            except (json.JSONDecodeError, KeyError) as error:
                raise RuntimeError(
                    f"Invalid run manifest record at line {line_number}"
                ) from error
            if clip_id in completed:
                raise RuntimeError(f"Duplicate clip in run manifest: {clip_id}")
            completed[clip_id] = record

    return manifest_path, completed


def validate_completed_clips(
    output_dir: Path,
    clip_plan: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    locked_config: dict[str, Any],
) -> None:
    planned = {clip["clip_id"]: clip for clip in clip_plan}
    unexpected = set(completed) - set(planned)
    if unexpected:
        raise RuntimeError(f"Run manifest contains unexpected clips: {sorted(unexpected)}")

    for clip_id, record in completed.items():
        expected = {
            **planned[clip_id],
            "model_source": locked_config["model_source"],
            "duration_seconds": locked_config["generation"]["duration"],
            **clip_backend_metadata(locked_config),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise RuntimeError(f"Run manifest metadata mismatch for {clip_id}: {key}")
        if not isinstance(record.get("sample_rate"), int) or record["sample_rate"] <= 0:
            raise RuntimeError(f"Invalid sample rate in run manifest for {clip_id}")
        audio_path = output_dir / record["audio_path"]
        if not audio_path.is_file():
            raise RuntimeError(f"Completed clip is missing its audio file: {audio_path}")


def package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def seed_everything(torch: Any, seed: int, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def generate_musicgen(
    args: argparse.Namespace,
    output_dir: Path,
    model_source: str,
    locked_config: dict[str, Any],
    clip_plan: list[dict[str, Any]],
) -> None:
    try:
        import torch
        from audiocraft.data.audio import audio_write
        from audiocraft.models.musicgen import MusicGen
    except ImportError as error:
        raise RuntimeError(
            "Generation requires the AudioCraft environment from this repository's "
            "Docker image."
        ) from error

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    manifest_path, completed = prepare_output(output_dir, locked_config)
    validate_completed_clips(output_dir, clip_plan, completed, locked_config)

    pending = [clip for clip in clip_plan if clip["clip_id"] not in completed]
    if not pending:
        print(f"run already complete: {len(completed)} clips in {output_dir}")
        return
    generation_batches = build_generation_batches(clip_plan, args.batch_size)
    pending_batches = [
        (batch, [clip for clip in batch if clip["clip_id"] not in completed])
        for batch in generation_batches
    ]
    pending_batches = [
        (batch, pending_batch)
        for batch, pending_batch in pending_batches
        if pending_batch
    ]

    print(f"loading {args.model} on {device}...")
    model = load_musicgen_model(
        MusicGen,
        torch,
        model_source,
        locked_config["model_source"],
        device,
    )
    if locked_config["model_source"]["type"] == "lora_adapter":
        projection_count = apply_adapter_scale(
            model.lm,
            locked_config["adapter_scale"],
        )
        print(
            f"applied adapter scale {locked_config['adapter_scale']} "
            f"to {projection_count} LoRA projections"
        )
    model.set_generation_params(**locked_config["generation"])

    runtime_path = output_dir / "runtime.json"
    runtime = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "musicgen",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "audiocraft": package_version("audiocraft"),
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        "sample_rate": model.sample_rate,
    }
    if runtime_path.exists():
        try:
            existing_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid runtime metadata: {runtime_path}") from error
        comparable_keys = set(runtime) - {"started_at_utc"}
        if any(existing_runtime.get(key) != runtime[key] for key in comparable_keys):
            raise RuntimeError(
                f"Runtime environment differs from {runtime_path}; use a new run name."
            )
    else:
        write_json(runtime_path, runtime)

    print(
        f"generating {len(pending)} of {len(clip_plan)} clips "
        f"in {len(pending_batches)} batches into {output_dir / 'audio'}"
    )
    generated_count = 0
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for batch_index, (batch, pending_batch) in enumerate(
            pending_batches, start=1
        ):
            for clip in pending_batch:
                output_path = output_dir / clip["audio_path"]
                if output_path.exists():
                    raise RuntimeError(
                        f"Refusing to overwrite untracked audio file: {output_path}"
                    )

            seed = batch[0]["seed"]
            print(
                f"[batch {batch_index}/{len(pending_batches)}] seed {seed}: "
                f"{len(pending_batch)} pending of {len(batch)} clips"
            )
            seed_everything(torch, seed, device)
            with torch.inference_mode():
                waveforms = model.generate(
                    [clip["prompt"] for clip in batch], progress=True
                )
            if len(waveforms) != len(batch):
                raise RuntimeError(
                    "AudioCraft returned an unexpected number of waveforms: "
                    f"expected {len(batch)}, got {len(waveforms)}"
                )

            for clip, waveform in zip(batch, waveforms):
                if clip["clip_id"] in completed:
                    continue
                output_path = output_dir / clip["audio_path"]
                audio_write(
                    str(output_path.with_suffix("")),
                    waveform.cpu(),
                    model.sample_rate,
                    **AUDIO_WRITE_PARAMS,
                )
                if not output_path.is_file():
                    raise RuntimeError(
                        f"AudioCraft did not create expected file: {output_path}"
                    )

                record = {
                    **clip,
                    "model_source": locked_config["model_source"],
                    **clip_backend_metadata(locked_config),
                    "sample_rate": model.sample_rate,
                    "duration_seconds": locked_config["generation"]["duration"],
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_file.flush()
                os.fsync(manifest_file.fileno())
                generated_count += 1
                print(
                    f"[{generated_count}/{len(pending)}] {clip['prompt_id']} "
                    f"(seed {clip['seed']})"
                )

    print(f"run complete: {len(clip_plan)} clips in {output_dir}")


def resolve_ace_step_dtype(torch: Any, requested: str, device: str) -> Any:
    dtype_name = requested
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device.startswith("cuda") else "float32"
    if not device.startswith("cuda") and dtype_name == "float16":
        raise RuntimeError(
            "ACE-Step float16 inference requires CUDA; use --ace-dtype float32 "
            "or bfloat16 on CPU."
        )
    return getattr(torch, dtype_name)


def normalize_ace_step_audio(
    torch: Any,
    torchaudio: Any,
    waveform: Any,
    sample_rate: int,
) -> Any:
    energy = waveform.pow(2).mean().sqrt().item()
    if energy >= 2e-3:
        input_loudness_db = torchaudio.transforms.Loudness(sample_rate)(
            waveform
        ).item()
        gain = 10.0 ** ((-14.0 - input_loudness_db) / 20.0)
        waveform = torch.tanh(gain * waveform)
    waveform = waveform.clamp(-1, 1)
    if not waveform.isfinite().all():
        raise RuntimeError("ACE-Step returned non-finite audio samples")
    return waveform


def load_ace_step_pipeline(
    AceStepPipeline: Any,
    torch: Any,
    model_source: str,
    model_source_record: dict[str, Any],
    device: str,
    requested_dtype: str,
    cpu_offload: bool,
) -> tuple[Any, Any]:
    installed_diffusers = package_version("diffusers")
    if installed_diffusers != ACE_STEP_DIFFUSERS_VERSION:
        raise RuntimeError(
            f"ACE-Step generation requires diffusers=={ACE_STEP_DIFFUSERS_VERSION}; "
            f"found {installed_diffusers!r}."
        )
    dtype = resolve_ace_step_dtype(torch, requested_dtype, device)
    pipeline = AceStepPipeline.from_pretrained(
        model_source,
        revision=model_source_record["revision"],
        torch_dtype=dtype,
    )
    if cpu_offload:
        if not device.startswith("cuda"):
            raise RuntimeError("--ace-cpu-offload requires a CUDA device")
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline = pipeline.to(device)
    pipeline.vae.enable_tiling()
    return pipeline, dtype


def generate_ace_step(
    args: argparse.Namespace,
    output_dir: Path,
    model_source: str,
    locked_config: dict[str, Any],
    clip_plan: list[dict[str, Any]],
) -> None:
    try:
        import soundfile
        import torch
        import torchaudio
        from diffusers import AceStepPipeline
    except ImportError as error:
        raise RuntimeError(
            "ACE-Step generation requires a separate Python environment "
            "installed from ace-step-requirements.txt."
        ) from error

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    manifest_path, completed = prepare_output(output_dir, locked_config)
    validate_completed_clips(output_dir, clip_plan, completed, locked_config)

    pending = [clip for clip in clip_plan if clip["clip_id"] not in completed]
    if not pending:
        print(f"run already complete: {len(completed)} clips in {output_dir}")
        return
    generation_batches = build_generation_batches(clip_plan, args.batch_size)
    pending_batches = [
        (batch, [clip for clip in batch if clip["clip_id"] not in completed])
        for batch in generation_batches
    ]
    pending_batches = [
        (batch, pending_batch)
        for batch, pending_batch in pending_batches
        if pending_batch
    ]

    print(f"loading {args.model} on {device}...")
    pipeline, dtype = load_ace_step_pipeline(
        AceStepPipeline,
        torch,
        model_source,
        locked_config["model_source"],
        device,
        locked_config["ace_dtype"],
        locked_config["ace_cpu_offload"],
    )
    sample_rate = pipeline.sample_rate

    runtime_path = output_dir / "runtime.json"
    runtime = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "ace-step",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda": torch.version.cuda,
        "diffusers": package_version("diffusers"),
        "transformers": package_version("transformers"),
        "soundfile": package_version("soundfile"),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        "dtype": str(dtype).removeprefix("torch."),
        "cpu_offload": locked_config["ace_cpu_offload"],
        "sample_rate": sample_rate,
    }
    if runtime_path.exists():
        try:
            existing_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid runtime metadata: {runtime_path}") from error
        comparable_keys = set(runtime) - {"started_at_utc"}
        if any(existing_runtime.get(key) != runtime[key] for key in comparable_keys):
            raise RuntimeError(
                f"Runtime environment differs from {runtime_path}; use a new run name."
            )
    else:
        write_json(runtime_path, runtime)

    generation = locked_config["generation"]
    print(
        f"generating {len(pending)} of {len(clip_plan)} clips "
        f"in {len(pending_batches)} batches into {output_dir / 'audio'}"
    )
    generated_count = 0
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for batch_index, (batch, pending_batch) in enumerate(
            pending_batches, start=1
        ):
            for clip in pending_batch:
                output_path = output_dir / clip["audio_path"]
                if output_path.exists():
                    raise RuntimeError(
                        f"Refusing to overwrite untracked audio file: {output_path}"
                    )

            seed = batch[0]["seed"]
            print(
                f"[batch {batch_index}/{len(pending_batches)}] seed {seed}: "
                f"{len(pending_batch)} pending of {len(batch)} clips"
            )
            seed_everything(torch, seed, device)
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                result = pipeline(
                    prompt=[clip["prompt"] for clip in batch],
                    lyrics=[generation["lyrics"]] * len(batch),
                    audio_duration=generation["duration"],
                    task_type=generation["task_type"],
                    num_inference_steps=generation["num_inference_steps"],
                    guidance_scale=generation["guidance_scale"],
                    shift=generation["shift"],
                    generator=generator,
                    output_type="pt",
                )
            waveforms = result.audios
            if len(waveforms) != len(batch):
                raise RuntimeError(
                    "ACE-Step returned an unexpected number of waveforms: "
                    f"expected {len(batch)}, got {len(waveforms)}"
                )

            for clip, waveform in zip(batch, waveforms):
                if clip["clip_id"] in completed:
                    continue
                output_path = output_dir / clip["audio_path"]
                waveform = waveform.detach().cpu().float()
                if waveform.ndim != 2:
                    raise RuntimeError(
                        "ACE-Step returned a waveform with an unexpected shape: "
                        f"{tuple(waveform.shape)}"
                    )
                waveform = normalize_ace_step_audio(
                    torch,
                    torchaudio,
                    waveform,
                    sample_rate,
                )
                soundfile.write(
                    str(output_path),
                    waveform.transpose(0, 1).numpy(),
                    sample_rate,
                    format="WAV",
                    subtype=locked_config["audio_write"]["subtype"],
                )
                if not output_path.is_file():
                    raise RuntimeError(
                        f"ACE-Step did not create expected file: {output_path}"
                    )

                record = {
                    **clip,
                    "model_source": locked_config["model_source"],
                    **clip_backend_metadata(locked_config),
                    "sample_rate": sample_rate,
                    "duration_seconds": generation["duration"],
                }
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_file.flush()
                os.fsync(manifest_file.fileno())
                generated_count += 1
                print(
                    f"[{generated_count}/{len(pending)}] {clip['prompt_id']} "
                    f"(seed {clip['seed']})"
                )

    print(f"run complete: {len(clip_plan)} clips in {output_dir}")


def generate(
    args: argparse.Namespace,
    output_dir: Path,
    model_source: str,
    locked_config: dict[str, Any],
    clip_plan: list[dict[str, Any]],
) -> None:
    backend = locked_config.get("backend", DEFAULT_BACKEND)
    if backend == "musicgen":
        generate_musicgen(
            args,
            output_dir,
            model_source,
            locked_config,
            clip_plan,
        )
    elif backend == "ace-step":
        generate_ace_step(
            args,
            output_dir,
            model_source,
            locked_config,
            clip_plan,
        )
    else:
        raise RuntimeError(f"Unsupported generation backend: {backend!r}")


def main() -> None:
    args = parse_args()
    prompt_sha256 = verify_prompt_checksum()
    prompts = load_prompts()
    output_dir = validate_args(args, len(prompts))
    if args.limit is not None:
        prompts = prompts[: args.limit]

    model_source, model_source_record = resolve_model_source(
        args.model,
        args.backend,
        args.model_revision,
    )
    locked_config = build_locked_config(
        args, prompts, prompt_sha256, model_source_record
    )
    clip_plan = build_clip_plan(prompts, args.seeds)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "clip_count": len(clip_plan),
                    "locked_config": locked_config,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    generate(
        args,
        output_dir,
        model_source,
        locked_config,
        clip_plan,
    )


if __name__ == "__main__":
    main()
