import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0"


def write_dataset_manifest(output_dir: Path) -> Path:
    """Write SHA-256 integrity metadata for all generated dataset artifacts."""
    manifest_path = output_dir / DATASET_MANIFEST_FILENAME
    entries = [
        _file_entry(output_dir, path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_dataset_manifest(output_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Verify that every artifact recorded in a dataset manifest is unchanged."""
    manifest_path = output_dir / DATASET_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return [f"Missing dataset manifest: {DATASET_MANIFEST_FILENAME}"], {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid dataset manifest: {exc}"], {}
    if not isinstance(payload, dict) or payload.get("algorithm") != "sha256":
        return ["Dataset manifest must use the sha256 algorithm."], {}
    entries = payload.get("files")
    if not isinstance(entries, list):
        return ["Dataset manifest field 'files' must be a list."], {}

    errors: list[str] = []
    recorded_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("Dataset manifest contains an invalid file entry.")
            continue
        relative_path = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("bytes")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str) or not isinstance(expected_size, int):
            errors.append("Dataset manifest contains an incomplete file entry.")
            continue
        if relative_path in recorded_paths:
            errors.append(f"Dataset manifest records a duplicate path: {relative_path}")
            continue
        recorded_paths.add(relative_path)
        path = output_dir / relative_path
        if not path.is_file():
            errors.append(f"Dataset manifest references a missing file: {relative_path}")
            continue
        if path.stat().st_size != expected_size:
            errors.append(f"Dataset manifest size mismatch: {relative_path}")
        if _sha256(path) != expected_hash:
            errors.append(f"Dataset manifest hash mismatch: {relative_path}")

    return errors, {
        "schema_version": payload.get("manifest_schema_version"),
        "algorithm": payload.get("algorithm"),
        "file_count": len(entries),
    }


def _file_entry(output_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
