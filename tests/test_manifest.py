import json
from pathlib import Path

from src.core.config import DatasetConfig
from src.core.generator import generate_dataset
from src.core.manifest import DATASET_MANIFEST_FILENAME, verify_dataset_manifest
from src.core.quality import validate_dataset


def test_generated_dataset_contains_verifiable_manifest_and_config_version(tmp_path: Path) -> None:
    output_dir = generate_dataset(
        DatasetConfig(
            output_dir=tmp_path / "dataset",
            classes=("circle", "square"),
            train_count=1,
            val_count=1,
            test_count=1,
            seed=29,
        )
    )

    manifest_path = output_dir / DATASET_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((output_dir / "generation_config.json").read_text(encoding="utf-8"))
    errors, statistics = verify_dataset_manifest(output_dir)

    assert manifest["algorithm"] == "sha256"
    assert manifest["files"]
    assert config["config_version"] == "1.0"
    assert errors == []
    assert statistics["file_count"] == len(manifest["files"])
    assert validate_dataset(output_dir, DatasetConfig(
        output_dir=output_dir,
        classes=("circle", "square"),
        train_count=1,
        val_count=1,
        test_count=1,
        seed=29,
    )).passed


def test_manifest_detects_modified_generated_file(tmp_path: Path) -> None:
    output_dir = generate_dataset(
        DatasetConfig(
            output_dir=tmp_path / "dataset",
            classes=("circle",),
            train_count=1,
            val_count=1,
            test_count=1,
        )
    )
    image_path = next((output_dir / "train" / "circle").glob("*.png"))
    image_path.write_bytes(b"modified")

    errors, _ = verify_dataset_manifest(output_dir)

    assert any("hash mismatch" in error for error in errors)
