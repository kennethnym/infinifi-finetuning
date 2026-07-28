import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


AUDIOCRAFT_COMMIT = "adf0b04a4452f171970028fcf80f101dd5e26e19"
BASE_MODEL = "facebook/musicgen-small"
PROJECT_ROOT = Path(__file__).resolve().parent
AUDIOCRAFT_PATCH = PROJECT_ROOT / "patches" / "audiocraft-lora.patch"
ADAPTER_METADATA = "adapter.json"
ADAPTER_WEIGHTS = "adapter_state.bin"
SAFE_SIGNATURE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SUPPORTED_TARGETS = {
    "self_attention",
    "cross_attention",
    "feedforward",
}


def parse_checkpoint(value: str) -> int | None:
    if value == "latest":
        return None
    if value.isdigit() and int(value) > 0:
        return int(value)
    raise argparse.ArgumentTypeError(
        "--checkpoint must be 'latest' or a positive epoch number"
    )


def checkpoint_filename(checkpoint: int | None) -> str:
    return "checkpoint.th" if checkpoint is None else f"checkpoint_{checkpoint}.th"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Dora MusicGen LoRA checkpoint as an adapter package."
    )
    parser.add_argument(
        "--signature",
        required=True,
        help="Dora experiment signature containing the requested checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint,
        default=None,
        metavar="EPOCH",
        help=(
            "Checkpoint epoch to export, or 'latest' for checkpoint.th "
            "(default: latest)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for the exported adapter package.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def nested_value(value: Any, *path: str) -> Any:
    for part in path:
        if isinstance(value, Mapping):
            if part not in value:
                raise RuntimeError(f"Checkpoint configuration is missing {'.'.join(path)}")
            value = value[part]
        else:
            try:
                value = getattr(value, part)
            except AttributeError as error:
                raise RuntimeError(
                    f"Checkpoint configuration is missing {'.'.join(path)}"
                ) from error
    return value


def normalize_lora_config(checkpoint_package: Mapping[str, Any]) -> dict[str, Any]:
    if "xp.cfg" not in checkpoint_package:
        raise RuntimeError("Checkpoint does not contain xp.cfg")
    raw = nested_value(checkpoint_package["xp.cfg"], "transformer_lm", "lora")
    try:
        config = {
            "enabled": bool(nested_value(raw, "enabled")),
            "rank": int(nested_value(raw, "rank")),
            "alpha": float(nested_value(raw, "alpha")),
            "dropout": float(nested_value(raw, "dropout")),
            "targets": list(nested_value(raw, "targets")),
        }
    except (TypeError, ValueError) as error:
        raise RuntimeError("Checkpoint contains an invalid LoRA configuration") from error

    if not config["enabled"]:
        raise RuntimeError("Checkpoint LoRA configuration is disabled")
    if config["rank"] <= 0:
        raise RuntimeError("Checkpoint LoRA rank must be positive")
    if not math.isfinite(config["alpha"]) or config["alpha"] <= 0:
        raise RuntimeError("Checkpoint LoRA alpha must be finite and positive")
    if not math.isfinite(config["dropout"]) or not 0 <= config["dropout"] < 1:
        raise RuntimeError("Checkpoint LoRA dropout must be in [0, 1)")
    unknown_targets = set(config["targets"]) - SUPPORTED_TARGETS
    if not config["targets"] or unknown_targets:
        raise RuntimeError(
            f"Checkpoint contains invalid LoRA targets: {config['targets']}"
        )
    return config


def is_lora_key(name: str) -> bool:
    return any(part.startswith("lora_") for part in name.split("."))


def extract_adapter_state(
    checkpoint_package: Mapping[str, Any],
) -> dict[str, Any]:
    fsdp_best_state = checkpoint_package.get("fsdp_best_state")
    if fsdp_best_state:
        raise RuntimeError("FSDP LoRA checkpoints are not supported")
    try:
        model_state = checkpoint_package["best_state"]["model"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Checkpoint does not contain best_state.model") from error
    if not isinstance(model_state, Mapping):
        raise RuntimeError("Checkpoint best_state.model is not a state dictionary")
    adapter_state = {
        name: tensor
        for name, tensor in model_state.items()
        if is_lora_key(name)
    }
    if not adapter_state:
        raise RuntimeError("Checkpoint contains no LoRA adapter tensors")
    return adapter_state


def parameter_count(adapter_state: Mapping[str, Any]) -> int:
    total = 0
    for name, tensor in adapter_state.items():
        numel = getattr(tensor, "numel", None)
        if numel is None:
            raise RuntimeError(f"Adapter state value is not a tensor: {name}")
        total += int(numel())
    return total


def expected_metadata(
    signature: str,
    checkpoint_name: str,
    checkpoint_sha256: str,
    lora_config: dict[str, Any],
    trainable_parameters: int,
    weights_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": "infinifi_musicgen_lora",
        "base_model": BASE_MODEL,
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
        "audiocraft_patch_sha256": sha256_file(AUDIOCRAFT_PATCH),
        "signature": signature,
        "source_checkpoint": {
            "filename": checkpoint_name,
            "sha256": checkpoint_sha256,
        },
        "lora": lora_config,
        "gate": {
            "source": "condition_mask",
            "active_when": "any_condition_token",
        },
        "trainable_parameters": trainable_parameters,
        "files": {
            ADAPTER_WEIGHTS: file_metadata(weights_path),
        },
    }


def validate_existing_export(
    output_dir: Path,
    signature: str,
    checkpoint_name: str,
) -> tuple[bool, str]:
    expected_paths = {ADAPTER_METADATA, ADAPTER_WEIGHTS}
    actual_paths = set()
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            return False, f"contains a symlink: {path}"
        if path.is_file():
            actual_paths.add(path.relative_to(output_dir).as_posix())
    if actual_paths != expected_paths:
        return (
            False,
            f"expected exactly {sorted(expected_paths)}, found {sorted(actual_paths)}",
        )

    metadata_path = output_dir / ADAPTER_METADATA
    try:
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"cannot read {metadata_path}: {error}"
    fixed = {
        "schema_version": 1,
        "format": "infinifi_musicgen_lora",
        "base_model": BASE_MODEL,
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
        "audiocraft_patch_sha256": sha256_file(AUDIOCRAFT_PATCH),
        "signature": signature,
    }
    for key, expected in fixed.items():
        if recorded.get(key) != expected:
            return False, f"{key} does not match the requested export"
    source = recorded.get("source_checkpoint")
    if not isinstance(source, dict) or source.get("filename") != checkpoint_name:
        return False, "source checkpoint does not match the requested export"
    recorded_file = recorded.get("files", {}).get(ADAPTER_WEIGHTS)
    try:
        actual_file = file_metadata(output_dir / ADAPTER_WEIGHTS)
    except OSError as error:
        return False, f"cannot hash adapter weights: {error}"
    if recorded_file != actual_file:
        return False, "recorded adapter weight hash or size does not match"
    return True, ""


def export_adapter(
    signature: str,
    output_dir: Path,
    checkpoint: int | None = None,
) -> None:
    if not SAFE_SIGNATURE.fullmatch(signature):
        raise RuntimeError(
            "--signature must start with an ASCII letter or digit, contain only "
            "letters, digits, '.', '_', or '-', and be at most 128 characters."
        )
    invalid_checkpoint = checkpoint is not None and (
        isinstance(checkpoint, bool)
        or not isinstance(checkpoint, int)
        or checkpoint <= 0
    )
    if invalid_checkpoint:
        raise RuntimeError("checkpoint must be a positive epoch number")
    checkpoint_name = checkpoint_filename(checkpoint)

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        compatible, reason = validate_existing_export(
            output_dir, signature, checkpoint_name
        )
        if compatible:
            print(f"compatible adapter export already exists: {output_dir}")
            return
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory {output_dir}: {reason}"
        )

    try:
        import torch
        from audiocraft import train
    except ImportError as error:
        raise RuntimeError(
            "Adapter export requires the patched AudioCraft environment from "
            "this repository's Docker image."
        ) from error

    experiment = train.main.get_xp_from_sig(signature)
    checkpoint_path = experiment.folder / checkpoint_name
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Dora checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_package = torch.load(checkpoint_path, map_location="cpu")
    lora_config = normalize_lora_config(checkpoint_package)
    adapter_state = extract_adapter_state(checkpoint_package)
    trainable_parameters = parameter_count(adapter_state)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.adapter-export-",
            dir=output_dir.parent,
        )
    )
    try:
        weights_path = staging_dir / ADAPTER_WEIGHTS
        torch.save(
            {
                "format": "infinifi_musicgen_lora",
                "state_dict": adapter_state,
            },
            weights_path,
        )
        if not weights_path.is_file() or weights_path.is_symlink():
            raise RuntimeError(f"Adapter export did not create expected file: {weights_path}")
        metadata_value = expected_metadata(
            signature,
            checkpoint_name,
            checkpoint_sha256,
            lora_config,
            trainable_parameters,
            weights_path,
        )
        (staging_dir / ADAPTER_METADATA).write_text(
            json.dumps(metadata_value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if output_dir.exists():
            if not output_dir.is_dir() or any(output_dir.iterdir()):
                raise RuntimeError(
                    f"Output directory changed during export; refusing to replace it: "
                    f"{output_dir}"
                )
            output_dir.rmdir()
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(
        f"exported {checkpoint_name} from Dora signature {signature} "
        f"to adapter package {output_dir}"
    )


def main() -> None:
    args = parse_args()
    export_adapter(args.signature, args.output_dir, args.checkpoint)


if __name__ == "__main__":
    main()
