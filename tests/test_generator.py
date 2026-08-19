from pathlib import Path
import csv

import numpy as np
import pytest
from PIL import Image

from src.core import generator
from src.core.config import SUPPORTED_COLOR_MODES, SUPPORTED_FILL_MODES, DatasetConfig, ShapeRenderConfig
from src.core.generator import (
    SUPPORTED_IMAGE_FORMATS,
    clear_dataset_dir,
    generate_dataset,
    validate_generated_dataset_dir,
)
from src.core.preview import generate_preview_images
from src.core.shapes import render_shape


def test_generate_dataset_creates_expected_structure(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "shapes",
        classes=("circle", "octagon", "square"),
        train_count=2,
        test_count=1,
        seed=7,
    )

    output_dir = generate_dataset(config)

    assert (output_dir / "train" / "circle").is_dir()
    assert (output_dir / "train" / "octagon").is_dir()
    assert (output_dir / "train" / "square").is_dir()
    assert (output_dir / "val" / "circle").is_dir()
    assert (output_dir / "val" / "octagon").is_dir()
    assert (output_dir / "val" / "square").is_dir()
    assert (output_dir / "test" / "circle").is_dir()
    assert (output_dir / "test" / "octagon").is_dir()
    assert (output_dir / "test" / "square").is_dir()
    assert len(list((output_dir / "train" / "circle").glob("*.png"))) == 2
    assert len(list((output_dir / "train" / "octagon").glob("*.png"))) == 2
    assert len(list((output_dir / "val" / "circle").glob("*.png"))) == 10
    assert len(list((output_dir / "val" / "octagon").glob("*.png"))) == 10
    assert len(list((output_dir / "test" / "circle").glob("*.png"))) == 1
    assert len(list((output_dir / "test" / "octagon").glob("*.png"))) == 1
    assert (output_dir / "metadata.csv").is_file()
    assert (output_dir / "class_mapping.json").is_file()
    assert (output_dir / "generation_config.json").is_file()
    assert (output_dir / "log.txt").is_file()
    assert (output_dir / "README.md").is_file()


def test_metadata_records_provenance_rendering_and_geometry(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "metadata",
        classes=("circle", "rectangle", "triangle"),
        train_count=1,
        val_count=1,
        test_count=1,
        render=ShapeRenderConfig(
            fill_mode="filled",
            rotation_range=(-20.0, 20.0),
            edge_clipping=True,
            noise_level=3,
        ),
    )

    output_dir = generate_dataset(config)
    with (output_dir / "metadata.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 9
    for row in rows:
        assert row["metadata_schema_version"] == "2.1"
        assert row["application_version"]
        assert row["python_version"]
        assert row["numpy_version"]
        assert row["pillow_version"]
        assert row["color_mode"] == "grayscale"
        assert row["fill_mode"] == "filled"
        assert row["noise_level"] == "3"
        assert row["edge_clipping"] == "True"
        assert row["clipping_side"] in {"left", "right", "top", "bottom"}
        assert float(row["shape_width"]) > 0
        assert float(row["shape_height"]) > 0
        assert float(row["shape_area"]) > 0
        assert float(row["bbox_right"]) > float(row["bbox_left"])
        assert float(row["bbox_bottom"]) > float(row["bbox_top"])


def test_generate_dataset_optionally_creates_binary_masks(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "masks",
        classes=("circle", "square"),
        train_count=1,
        val_count=1,
        test_count=1,
        save_masks=True,
        render=ShapeRenderConfig(fill_mode="outline_fill", edge_clipping=True, noise_level=8),
    )

    output_dir = generate_dataset(config)
    with (output_dir / "metadata.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 6
    for row in rows:
        mask_path = output_dir / row["mask_filename"]
        assert mask_path.is_file()
        with Image.open(mask_path) as mask:
            assert mask.format == "PNG"
            assert mask.mode == "L"
            assert mask.size == (64, 64)
            assert set(mask.getdata()) <= {0, 255}
            assert 255 in set(mask.getdata())


def test_clear_dataset_dir_removes_current_dataset_contents(tmp_path: Path) -> None:
    output_dir = tmp_path / "shapes"
    train_dir = output_dir / "train" / "circle"
    val_dir = output_dir / "val" / "circle"
    test_dir = output_dir / "test" / "circle"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "sample.png").write_text("image placeholder", encoding="utf-8")
    (val_dir / "sample.png").write_text("image placeholder", encoding="utf-8")
    (test_dir / "sample.png").write_text("image placeholder", encoding="utf-8")
    (output_dir / "metadata.csv").write_text("filename,class\n", encoding="utf-8")

    cleared_dir = clear_dataset_dir(output_dir)

    assert cleared_dir == output_dir
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_validate_generated_dataset_dir_rejects_folder_without_required_subsets(tmp_path: Path) -> None:
    output_dir = tmp_path / "documents"
    output_dir.mkdir()
    (output_dir / "notes.txt").write_text("not a dataset", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing subset"):
        validate_generated_dataset_dir(output_dir)


def test_validate_generated_dataset_dir_rejects_folder_without_validation_subset(tmp_path: Path) -> None:
    output_dir = tmp_path / "incomplete-dataset"
    train_dir = output_dir / "train" / "circle"
    test_dir = output_dir / "test" / "circle"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "sample.png").write_text("image placeholder", encoding="utf-8")
    (test_dir / "sample.png").write_text("image placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="val"):
        validate_generated_dataset_dir(output_dir)


def test_validate_generated_dataset_dir_rejects_folder_without_images(tmp_path: Path) -> None:
    output_dir = tmp_path / "shapes"
    (output_dir / "train" / "circle").mkdir(parents=True)
    (output_dir / "val" / "circle").mkdir(parents=True)
    (output_dir / "test" / "circle").mkdir(parents=True)

    with pytest.raises(ValueError, match="No image files"):
        validate_generated_dataset_dir(output_dir)


@pytest.mark.parametrize("file_format", SUPPORTED_IMAGE_FORMATS)
def test_generate_dataset_supports_standard_image_formats(tmp_path: Path, file_format: str) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / file_format,
        classes=("circle",),
        train_count=1,
        test_count=1,
        file_format=file_format,
        seed=3,
    )

    output_dir = generate_dataset(config)

    assert len(list(output_dir.glob(f"train/circle/*.{file_format}"))) == 1
    assert len(list(output_dir.glob(f"test/circle/*.{file_format}"))) == 1


def test_generate_dataset_rejects_unsupported_image_format(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "bad-format",
        classes=("circle",),
        train_count=1,
        test_count=1,
        file_format="txt",
    )

    with pytest.raises(ValueError, match="Unsupported image format"):
        generate_dataset(config)


def test_generate_dataset_rejects_center_offset_larger_than_half_image_size(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "bad-offset",
        image_size=64,
        classes=("circle",),
        train_count=1,
        test_count=1,
        render=ShapeRenderConfig(center_offset_range=(0.0, 33.0)),
    )

    with pytest.raises(ValueError, match="half the image size"):
        generate_dataset(config)

    assert not (tmp_path / "bad-offset").exists()


def test_generate_dataset_cleans_up_after_interrupted_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "interrupted"
    config = DatasetConfig(
        output_dir=output_dir,
        classes=("circle",),
        train_count=2,
        val_count=1,
        test_count=1,
    )

    def interrupt_generation(*_args: object) -> None:
        raise RuntimeError("generation interrupted")

    with pytest.raises(RuntimeError, match="generation interrupted"):
        generate_dataset(config, progress_callback=interrupt_generation)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_generate_preview_images_does_not_create_dataset(tmp_path: Path) -> None:
    output_dir = tmp_path / "preview-only"
    config = DatasetConfig(
        output_dir=output_dir,
        classes=("circle", "square"),
        train_count=50,
        test_count=10,
        seed=11,
    )

    previews = generate_preview_images(config, limit=5)

    assert [preview.class_name for preview in previews] == ["circle", "square", "circle", "square", "circle"]
    assert all(preview.image.size == (64, 64) for preview in previews)
    assert not output_dir.exists()


def test_generate_preview_images_is_reproducible() -> None:
    config = DatasetConfig(classes=("triangle",), seed=17)

    first = generate_preview_images(config, limit=1)[0].image
    second = generate_preview_images(config, limit=1)[0].image

    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_generate_dataset_supports_fill_rotation_and_scale(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "render-options",
        classes=("rectangle", "triangle"),
        train_count=1,
        test_count=1,
        seed=13,
        render=ShapeRenderConfig(
            fill_mode="filled",
            rotation_range=(-45.0, 45.0),
            scale_range=(0.5, 1.4),
            center_offset_range=(0.0, 12.0),
        ),
    )

    output_dir = generate_dataset(config)

    assert len(list(output_dir.glob("train/rectangle/*.png"))) == 1
    assert len(list(output_dir.glob("train/triangle/*.png"))) == 1


def test_generate_dataset_supports_rgb_mode(tmp_path: Path) -> None:
    config = DatasetConfig(
        output_dir=tmp_path / "rgb",
        classes=("square",),
        train_count=1,
        test_count=1,
        seed=21,
        render=ShapeRenderConfig(
            color_mode="rgb",
            background_color=(120, 130, 140),
            foreground_color=(10, 20, 30),
            fill_mode="filled",
        ),
    )

    output_dir = generate_dataset(config)
    image_path = next((output_dir / "train" / "square").glob("*.png"))

    with Image.open(image_path) as image:
        assert image.mode == "RGB"


@pytest.mark.parametrize("fill_mode", SUPPORTED_FILL_MODES)
def test_render_shape_supports_fill_modes(fill_mode: str) -> None:
    config = ShapeRenderConfig(
        background_range=(180, 181),
        foreground_range=(0, 1),
        line_width_range=(1, 2),
        fill_mode=fill_mode,
        rotation_range=(-30.0, 30.0),
        scale_range=(0.8, 1.2),
    )

    image = render_shape("square", np.random.default_rng(4), 64, config)

    assert image.mode == "L"
    assert image.size == (64, 64)


@pytest.mark.parametrize("color_mode", SUPPORTED_COLOR_MODES)
def test_render_shape_supports_color_modes(color_mode: str) -> None:
    config = ShapeRenderConfig(
        color_mode=color_mode,
        background_range=(180, 181),
        foreground_range=(0, 1),
        background_color=(120, 130, 140),
        foreground_color=(10, 20, 30),
        line_width_range=(1, 2),
        fill_mode="filled",
    )

    image = render_shape("octagon", np.random.default_rng(14), 64, config)

    assert image.mode == ("RGB" if color_mode == "rgb" else "L")
    assert image.size == (64, 64)


def test_render_shape_adds_background_noise_when_enabled() -> None:
    clean_config = ShapeRenderConfig(
        background_range=(180, 181),
        foreground_range=(0, 1),
        line_width_range=(1, 2),
        noise_level=0,
    )
    noisy_config = ShapeRenderConfig(
        background_range=(180, 181),
        foreground_range=(0, 1),
        line_width_range=(1, 2),
        noise_level=20,
    )

    clean = np.asarray(render_shape("circle", np.random.default_rng(9), 64, clean_config))
    noisy = np.asarray(render_shape("circle", np.random.default_rng(9), 64, noisy_config))

    assert len(np.unique(noisy)) > len(np.unique(clean))


def test_render_shape_supports_edge_clipping() -> None:
    config = ShapeRenderConfig(
        background_range=(200, 201),
        foreground_range=(0, 1),
        line_width_range=(2, 3),
        fill_mode="filled",
        edge_clipping=True,
        min_visible_ratio=0.6,
        scale_range=(1.2, 1.2),
    )

    image = render_shape("square", np.random.default_rng(8), 64, config)
    pixels = np.asarray(image)
    foreground = pixels < 50
    edge_pixels = np.concatenate(
        [foreground[0, :], foreground[-1, :], foreground[:, 0], foreground[:, -1]]
    )

    assert foreground.any()
    assert edge_pixels.any()


def test_shape_render_config_rejects_negative_noise_level() -> None:
    with pytest.raises(ValueError, match="Noise level"):
        ShapeRenderConfig(noise_level=-1)


def test_shape_render_config_rejects_invalid_fill_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported fill mode"):
        ShapeRenderConfig(fill_mode="invalid")


def test_shape_render_config_rejects_invalid_color_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported color mode"):
        ShapeRenderConfig(color_mode="cmyk")


def test_shape_render_config_rejects_invalid_rgb_component() -> None:
    with pytest.raises(ValueError, match="RGB components"):
        ShapeRenderConfig(background_color=(256, 180, 180))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("background_range", (1, 1), "Background intensity range"),
        ("foreground_range", (-1, 10), "Foreground intensity range"),
        ("line_width_range", (0, 1), "Line width range"),
        ("margin", -1, "Margin cannot be negative"),
    ],
)
def test_shape_render_config_rejects_invalid_sampling_ranges(
    field: str,
    value: tuple[int, int] | int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ShapeRenderConfig(**{field: value})


def test_shape_render_config_rejects_invalid_rotation_range() -> None:
    with pytest.raises(ValueError, match="Rotation minimum"):
        ShapeRenderConfig(rotation_range=(20.0, -20.0))


def test_shape_render_config_rejects_invalid_scale_range() -> None:
    with pytest.raises(ValueError, match="Scale minimum"):
        ShapeRenderConfig(scale_range=(1.5, 0.5))


def test_shape_render_config_rejects_negative_center_offset() -> None:
    with pytest.raises(ValueError, match="Center offset"):
        ShapeRenderConfig(center_offset_range=(-1.0, 5.0))


def test_shape_render_config_rejects_invalid_center_offset_range() -> None:
    with pytest.raises(ValueError, match="Center offset minimum"):
        ShapeRenderConfig(center_offset_range=(10.0, 5.0))


def test_shape_render_config_rejects_too_low_min_visible_ratio() -> None:
    with pytest.raises(ValueError, match="Minimum visible ratio"):
        ShapeRenderConfig(edge_clipping=True, min_visible_ratio=0.5)


def _low_variability_circle_config(output_dir: Path, total_per_subset: tuple[int, int, int]) -> DatasetConfig:
    train_count, val_count, test_count = total_per_subset
    return DatasetConfig(
        output_dir=output_dir,
        image_size=28,
        classes=("circle",),
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=31,
        render=ShapeRenderConfig(
            background_range=(100, 101),
            foreground_range=(0, 1),
            line_width_range=(1, 2),
            fill_mode="outline_fill",
        ),
    )


def test_generation_retries_until_images_are_unique_across_subsets(tmp_path: Path) -> None:
    dataset_dir = generate_dataset(_low_variability_circle_config(tmp_path / "unique", (1, 1, 1)))
    image_bytes = [path.read_bytes() for path in dataset_dir.glob("*/*/*.png")]

    assert len(image_bytes) == 3
    assert len(image_bytes) == len(set(image_bytes))


def test_generation_reports_when_configuration_cannot_produce_enough_unique_images(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generator, "MAX_DUPLICATE_ATTEMPTS", 3)

    with pytest.raises(ValueError, match="Unable to generate a unique image"):
        generate_dataset(_low_variability_circle_config(tmp_path / "exhausted", (2, 1, 1)))
