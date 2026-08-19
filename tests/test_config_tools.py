from pathlib import Path
import json

import pytest

from src.core.config import DatasetConfig, ShapeRenderConfig
from src.core.config_io import load_config_from_json, save_config_template
from src.core.generator import generate_dataset
from src.core.summary import dataset_summary, estimate_dataset_size
from src.core.validation import validate_config


def test_save_and_load_config_template_round_trips_dataset_config(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "dataset",
        image_size=128,
        classes=("circle", "triangle"),
        train_count=3,
        test_count=2,
        file_format="png",
        save_masks=True,
        seed=99,
        render=ShapeRenderConfig(
            fill_mode="outline_fill",
            rotation_range=(-15.0, 15.0),
            scale_range=(0.8, 1.2),
            center_offset_range=(0.0, 10.0),
            edge_clipping=True,
            min_visible_ratio=0.65,
            noise_level=4,
        ),
    )
    config_path = tmp_path / "generation_config.json"

    saved_path = save_config_template(config_path, config)
    loaded = load_config_from_json(saved_path)

    assert loaded == config


def test_load_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "not_a_generation_config.json"
    config_path.write_text('{"name": "not a ShapeDomainLab config"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required field"):
        load_config_from_json(config_path)


def test_load_config_without_version_uses_current_compatibility_version(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy-generation-config.json"
    save_config_template(config_path, DatasetConfig(output_dir=tmp_path / "dataset"))
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data.pop("config_version")
    config_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_config_from_json(config_path)

    assert loaded.config_version == DatasetConfig().config_version


def test_dataset_summary_reports_generated_counts(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "shapes",
        classes=("circle", "square"),
        train_count=2,
        test_count=1,
        seed=31,
    )
    output_dir = generate_dataset(config)

    summary = dataset_summary(output_dir)

    assert summary.exists
    assert summary.total_images == 26
    assert summary.subsets["train"] == {"circle": 2, "square": 2}
    assert summary.subsets["val"] == {"circle": 10, "square": 10}
    assert summary.subsets["test"] == {"circle": 1, "square": 1}
    assert summary.image_formats == {"png": 26}
    assert all(summary.service_files.values())


def test_dataset_summary_handles_missing_directory(tmp_path: Path) -> None:
    summary = dataset_summary(tmp_path / "missing")

    assert not summary.exists
    assert summary.total_images == 0


def test_estimate_dataset_size_uses_config_without_creating_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "estimate-only"
    config = DatasetConfig(
        output_dir=output_dir,
        classes=("circle", "octagon"),
        train_count=4,
        test_count=1,
        seed=2,
    )

    estimate = estimate_dataset_size(config)

    assert estimate.total_images == 30
    assert estimate.sample_count == 8
    assert estimate.estimated_total_bytes > 0
    assert not output_dir.exists()


def test_validate_config_reports_unsupported_classes_and_ranges() -> None:
    config = DatasetConfig(
        image_size=64,
        classes=("circle", "hexagon"),
        render=ShapeRenderConfig(center_offset_range=(0.0, 33.0)),
    )

    errors = validate_config(config)

    assert any("Unsupported shape class" in error for error in errors)
    assert any("half the image size" in error for error in errors)


def test_validate_config_requires_complete_integer_subset_totals() -> None:
    config = DatasetConfig(subset_totals={"train": 10, "val": "5"})

    errors = validate_config(config)

    assert any("Missing subset total(s): test" in error for error in errors)
    assert any("Total image count for val must be an integer" in error for error in errors)
