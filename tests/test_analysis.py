import json
from pathlib import Path

import pytest

from src.core.analysis import ANALYSIS_REPORT_JSON, ANALYSIS_REPORT_MARKDOWN, analyze_dataset
from src.core.config import DatasetConfig, ShapeRenderConfig
from src.core.generator import generate_dataset


def test_analyze_dataset_writes_machine_and_human_readable_reports(tmp_path: Path) -> None:
    dataset_dir = generate_dataset(
        DatasetConfig(
            output_dir=tmp_path / "dataset",
            classes=("circle", "square"),
            train_count=2,
            val_count=1,
            test_count=1,
            save_masks=True,
            seed=44,
            render=ShapeRenderConfig(
                fill_mode="outline_fill",
                rotation_range=(-15.0, 15.0),
                scale_range=(0.8, 1.2),
                noise_level=3,
            ),
        )
    )

    analysis_dir = analyze_dataset(dataset_dir, write_figures=False)
    report = json.loads((analysis_dir / ANALYSIS_REPORT_JSON).read_text(encoding="utf-8"))

    assert (analysis_dir / ANALYSIS_REPORT_MARKDOWN).is_file()
    assert report["sample_count"] == 8
    assert report["counts"]["by_subset_and_class"]["train"] == {"circle": 2, "square": 2}
    assert report["numeric_parameters"]["overall"]["scale"]["count"] == 8
    assert report["masks"]["enabled"]
    assert report["manifest"]["verified"]
    assert report["figures"] == []


def test_analyze_dataset_writes_figures_when_matplotlib_is_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    dataset_dir = generate_dataset(
        DatasetConfig(
            output_dir=tmp_path / "dataset",
            classes=("circle",),
            train_count=1,
            val_count=1,
            test_count=1,
            seed=8,
        )
    )

    analysis_dir = analyze_dataset(dataset_dir)
    report = json.loads((analysis_dir / ANALYSIS_REPORT_JSON).read_text(encoding="utf-8"))

    assert report["figures"]
    assert all((analysis_dir / figure).is_file() for figure in report["figures"])
