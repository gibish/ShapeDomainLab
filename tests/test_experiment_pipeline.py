import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.aggregate_results import collect_rows
from experiments.common import image_transform, metrics_from_predictions
from experiments.run_matrix import evaluation_complete, training_complete


def test_metrics_from_predictions_calculates_multiclass_summary() -> None:
    metrics = metrics_from_predictions([0, 0, 1, 1], [0, 1, 1, 1], ["circle", "square"])

    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["macro_f1"] > 0
    assert metrics["per_class"]["circle"]["support"] == 2


def test_collect_rows_reads_saved_metric_files(tmp_path: Path) -> None:
    run = tmp_path / "run-1"
    run.mkdir()
    (run / "metrics-rotation.json").write_text(
        json.dumps({"accuracy": 0.8, "balanced_accuracy": 0.75, "macro_f1": 0.7, "sample_count": 20}),
        encoding="utf-8",
    )

    rows = collect_rows(tmp_path)

    assert rows == [{"run": "run-1", "evaluation": "rotation", "accuracy": "0.8", "balanced_accuracy": "0.75", "macro_f1": "0.7", "sample_count": "20"}]


def test_image_transform_preserves_declared_grayscale_channel_count(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"render": {"color_mode": "grayscale"}}), encoding="utf-8"
    )
    pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    tensor = image_transform(tmp_path, torchvision)(Image.new("RGB", (8, 8), "white"))

    assert tuple(tensor.shape) == (1, 8, 8)


def test_matrix_stage_completion_requires_complete_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model.pt").write_bytes(b"checkpoint")
    (run_dir / "training_status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    assert training_complete(run_dir)

    (run_dir / "predictions-clean.csv").write_text("sample_index,target\n", encoding="utf-8")
    (run_dir / "metrics-clean.json").write_text(json.dumps({"macro_f1": 0.8}), encoding="utf-8")
    assert evaluation_complete(run_dir, "clean")

    (run_dir / "metrics-clean.json").write_text("incomplete", encoding="utf-8")
    assert not evaluation_complete(run_dir, "clean")
