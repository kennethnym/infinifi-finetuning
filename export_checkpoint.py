import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


AUDIOCRAFT_COMMIT = "adf0b04a4452f171970028fcf80f101dd5e26e19"
COMPRESSION_MODEL = "facebook/encodec_32khz"
EXPORT_METADATA = "export.json"
MODEL_FILES = ("state_dict.bin", "compression_state_dict.bin")
SAFE_SIGNATURE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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
        description="Export a Dora MusicGen checkpoint as a local AudioCraft package."
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
        help="New or empty directory for the exported model package.",
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


def expected_metadata(
    signature: str,
    output_dir: Path,
    checkpoint_name: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "signature": signature,
        "source_checkpoint": {
            "filename": checkpoint_name,
            "sha256": checkpoint_sha256,
        },
        "compression_model": COMPRESSION_MODEL,
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
        "files": {
            filename: file_metadata(output_dir / filename)
            for filename in MODEL_FILES
        },
    }


def validate_existing_export(
    output_dir: Path,
    signature: str,
    checkpoint_name: str,
) -> tuple[bool, str]:
    expected_paths = {*MODEL_FILES, EXPORT_METADATA}
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

    metadata_path = output_dir / EXPORT_METADATA
    try:
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"cannot read {metadata_path}: {error}"

    schema_version = recorded.get("schema_version")
    if schema_version == 1 and checkpoint_name == "checkpoint.th":
        fixed_fields = {
            "schema_version": 1,
            "signature": signature,
            "compression_model": COMPRESSION_MODEL,
            "audiocraft_commit": AUDIOCRAFT_COMMIT,
        }
        for key, expected_value in fixed_fields.items():
            if recorded.get(key) != expected_value:
                return False, f"{key} does not match the requested export"
        try:
            actual_files = {
                filename: file_metadata(output_dir / filename)
                for filename in MODEL_FILES
            }
        except OSError as error:
            return False, f"cannot hash exported model files: {error}"
        if recorded.get("files") != actual_files:
            return False, "recorded model file hashes or sizes do not match"
        return True, ""

    fixed_fields = {
        "schema_version": 2,
        "signature": signature,
        "compression_model": COMPRESSION_MODEL,
        "audiocraft_commit": AUDIOCRAFT_COMMIT,
    }
    for key, expected_value in fixed_fields.items():
        if recorded.get(key) != expected_value:
            return False, f"{key} does not match the requested export"

    source_checkpoint = recorded.get("source_checkpoint")
    if (
        not isinstance(source_checkpoint, dict)
        or source_checkpoint.get("filename") != checkpoint_name
    ):
        return False, "source checkpoint does not match the requested export"
    checkpoint_sha256 = source_checkpoint.get("sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        return False, "source checkpoint digest is invalid"

    try:
        actual_metadata = expected_metadata(
            signature,
            output_dir,
            checkpoint_name,
            checkpoint_sha256,
        )
    except OSError as error:
        return False, f"cannot hash exported model files: {error}"
    if recorded.get("files") != actual_metadata["files"]:
        return False, "recorded model file hashes or sizes do not match"
    return True, ""


def write_metadata(
    output_dir: Path,
    signature: str,
    checkpoint_name: str,
    checkpoint_sha256: str,
) -> None:
    metadata_path = output_dir / EXPORT_METADATA
    metadata_path.write_text(
        json.dumps(
            expected_metadata(
                signature,
                output_dir,
                checkpoint_name,
                checkpoint_sha256,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def export_checkpoint(
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
            output_dir,
            signature,
            checkpoint_name,
        )
        if compatible:
            print(f"compatible export already exists: {output_dir}")
            return
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory {output_dir}: {reason}"
        )

    try:
        from audiocraft import train
        from audiocraft.utils import export
    except ImportError as error:
        raise RuntimeError(
            "Checkpoint export requires the AudioCraft environment from this "
            "repository's Docker image."
        ) from error

    experiment = train.main.get_xp_from_sig(signature)
    checkpoint_path = experiment.folder / checkpoint_name
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Dora checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.export-",
            dir=output_dir.parent,
        )
    )
    try:
        export.export_lm(
            checkpoint_path,
            str(staging_dir / "state_dict.bin"),
        )
        export.export_pretrained_compression_model(
            COMPRESSION_MODEL,
            str(staging_dir / "compression_state_dict.bin"),
        )
        for filename in MODEL_FILES:
            path = staging_dir / filename
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"AudioCraft did not create expected file: {path}")
        write_metadata(
            staging_dir,
            signature,
            checkpoint_name,
            checkpoint_sha256,
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
        f"exported {checkpoint_name} from Dora signature {signature} to {output_dir}"
    )


def main() -> None:
    args = parse_args()
    export_checkpoint(args.signature, args.output_dir, args.checkpoint)


if __name__ == "__main__":
    main()
