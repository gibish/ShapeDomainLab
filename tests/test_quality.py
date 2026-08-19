from pathlib import Path
import json
import shutil

import pytest
from PIL import Image

from src.core.config import DatasetConfig, ShapeRenderConfig
from src.core.generator import generate_dataset
from src.core.quality import QUALITY_REPORT_JSON, QUALITY_REPORT_MARKDOWN, validate_dataset


def test_validate_dataset_passes_for_generated_dataset(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "shapes",
        classes=("circle", "octagon", "rectangle", "square", "triangle"),
        train_count=2,
        test_count=1,
        seed=11,
    )

    output_dir = generate_dataset(config)
    report = validate_dataset(output_dir, config)

    assert report.passed
    assert report.errors == []


def test_validate_dataset_passes_for_dataset_with_masks(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "masks",
        classes=("circle", "square"),
        train_count=1,
        val_count=1,
        test_count=1,
        save_masks=True,
        render=ShapeRenderConfig(fill_mode="outline_fill", noise_level=5),
    )

    output_dir = generate_dataset(config)
    report = validate_dataset(output_dir, config)

    assert report.passed
    assert report.statistics["masks"]["enabled"]
    assert report.statistics["masks"]["foreground_ratio"]["count"] == 6
    report_json = output_dir / QUALITY_REPORT_JSON
    assert report_json.is_file()
    assert (output_dir / QUALITY_REPORT_MARKDOWN).is_file()
    assert json.loads(report_json.read_text(encoding="utf-8"))["passed"]


def test_validate_dataset_detects_exact_duplicate_across_subsets(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "duplicates",
        classes=("circle",),
        train_count=1,
        val_count=1,
        test_count=1,
    )
    output_dir = generate_dataset(config)
    train_image = next((output_dir / "train" / "circle").glob("*.png"))
    test_image = next((output_dir / "test" / "circle").glob("*.png"))
    shutil.copyfile(train_image, test_image)

    report = validate_dataset(output_dir, config)

    assert not report.passed
    assert any("across dataset subsets" in error for error in report.errors)
    assert report.statistics["cross_subset_exact_duplicate_groups"]


@pytest.mark.parametrize("color_mode", ("grayscale", "rgb"))
def test_validate_dataset_accepts_palettized_gif_output(tmp_path: Path, color_mode: str) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / color_mode,
        classes=("circle",),
        train_count=1,
        val_count=1,
        test_count=1,
        file_format="gif",
        render=ShapeRenderConfig(color_mode=color_mode),
    )

    output_dir = generate_dataset(config)
    report = validate_dataset(output_dir, config)

    assert report.passed


def test_validate_dataset_rejects_unexpected_image_format(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "wrong-format",
        classes=("circle",),
        train_count=1,
        val_count=1,
        test_count=1,
    )
    output_dir = generate_dataset(config)
    image_path = next((output_dir / "train" / "circle").glob("*.png"))

    with Image.open(image_path) as image:
        image.save(image_path, format="JPEG")

    report = validate_dataset(output_dir, config)

    assert not report.passed
    assert any("Invalid image format" in error for error in report.errors)


def test_validate_dataset_rejects_unexpected_image_mode(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "wrong-mode",
        classes=("circle",),
        train_count=1,
        val_count=1,
        test_count=1,
    )
    output_dir = generate_dataset(config)
    image_path = next((output_dir / "train" / "circle").glob("*.png"))

    with Image.open(image_path) as image:
        image.convert("RGB").save(image_path, format="PNG")

    report = validate_dataset(output_dir, config)

    assert not report.passed
    assert any("Invalid image mode" in error for error in report.errors)
