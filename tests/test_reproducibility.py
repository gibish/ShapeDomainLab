import hashlib
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.core.config import DatasetConfig, ShapeRenderConfig
from src.core.generator import generate_dataset
from src.gui.app import MainWindow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dataset_hashes(dataset_dir: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )
    return {
        path.relative_to(dataset_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _small_config(output_dir: Path, seed: int) -> DatasetConfig:
    return DatasetConfig(
        output_dir=output_dir,
        classes=("circle", "square"),
        train_count=2,
        val_count=1,
        test_count=1,
        save_masks=True,
        seed=seed,
        render=ShapeRenderConfig(
            fill_mode="outline_fill",
            rotation_range=(-20.0, 20.0),
            scale_range=(0.8, 1.2),
            noise_level=5,
        ),
    )


def test_generation_is_byte_identical_for_the_same_seed(tmp_path: Path) -> None:
    first = generate_dataset(_small_config(tmp_path / "first", seed=73))
    second = generate_dataset(_small_config(tmp_path / "second", seed=73))

    assert _dataset_hashes(first) == _dataset_hashes(second)


def test_generation_changes_image_content_for_a_different_seed(tmp_path: Path) -> None:
    first = generate_dataset(_small_config(tmp_path / "first", seed=73))
    second = generate_dataset(_small_config(tmp_path / "second", seed=74))

    assert _dataset_hashes(first) != _dataset_hashes(second)


def test_cli_generates_and_summarizes_a_dataset(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-dataset"
    generation = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--output",
            str(output_dir),
            "--classes",
            "circle",
            "square",
            "--train-count",
            "1",
            "--val-count",
            "1",
            "--test-count",
            "1",
            "--save-masks",
            "--seed",
            "19",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "Dataset generated:" in generation.stdout
    assert "Quality check: passed" in generation.stdout
    assert (output_dir / "quality_report.json").is_file()

    summary = subprocess.run(
        [sys.executable, "main.py", "--summary", str(output_dir)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "Total images: 6" in summary.stdout
    assert "train: circle=1, square=1" in summary.stdout


def test_gui_main_window_can_be_constructed_headlessly() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.windowTitle() == "ShapeDomainLab"
    assert window.centralWidget() is not None

    window.close()
    application.processEvents()
