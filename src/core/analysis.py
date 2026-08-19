import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .manifest import verify_dataset_manifest


ANALYSIS_DIRECTORY = "analysis"
ANALYSIS_REPORT_JSON = "analysis_report.json"
ANALYSIS_REPORT_MARKDOWN = "analysis_report.md"
NUMERIC_FIELDS = ("scale", "rotation_degrees", "center_x", "center_y", "line_width", "shape_area")


def analyze_dataset(dataset_dir: Path | str, write_figures: bool = True) -> Path:
    """Create reproducible descriptive analysis artifacts for an existing dataset."""
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise ValueError(f"Dataset metadata file does not exist: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Dataset metadata contains no rows.")

    output_dir = dataset_dir / ANALYSIS_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = _numeric_statistics(rows)
    counts = _count_statistics(rows)
    masks = _mask_statistics(dataset_dir, rows)
    manifest_errors, manifest_statistics = verify_dataset_manifest(dataset_dir)
    quality = _quality_summary(dataset_dir)
    warnings = _confounder_warnings(numeric["by_class"])
    if manifest_errors:
        warnings.extend(manifest_errors)
    report: dict[str, Any] = {
        "dataset": str(dataset_dir),
        "sample_count": len(rows),
        "counts": counts,
        "numeric_parameters": numeric,
        "masks": masks,
        "manifest": {"verified": not manifest_errors, **manifest_statistics},
        "quality_report": quality,
        "potential_confounders": warnings,
        "figures": [],
    }
    if write_figures:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        report["figures"] = _write_figures(dataset_dir, rows, numeric, figures_dir)

    (output_dir / ANALYSIS_REPORT_JSON).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / ANALYSIS_REPORT_MARKDOWN).write_text(_markdown_report(report) + "\n", encoding="utf-8")
    return output_dir


def _count_statistics(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_subset_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_class: Counter[str] = Counter()
    for row in rows:
        subset = row.get("subset", "unknown")
        class_name = row.get("class", "unknown")
        by_subset_class[subset][class_name] += 1
        by_class[class_name] += 1
    return {
        "by_subset_and_class": {subset: dict(sorted(values.items())) for subset, values in sorted(by_subset_class.items())},
        "by_class": dict(sorted(by_class.items())),
    }


def _numeric_statistics(rows: list[dict[str, str]]) -> dict[str, Any]:
    overall: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_subset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for field in NUMERIC_FIELDS:
            value = _float_or_none(row.get(field))
            if value is None:
                continue
            overall[field].append(value)
            by_class[row.get("class", "unknown")][field].append(value)
            by_subset[row.get("subset", "unknown")][field].append(value)
    return {
        "overall": {field: _summary(values) for field, values in overall.items()},
        "by_class": _nested_summaries(by_class),
        "by_subset": _nested_summaries(by_subset),
    }


def _nested_summaries(values: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        group: {field: _summary(items) for field, items in sorted(fields.items())}
        for group, fields in sorted(values.items())
    }


def _mask_statistics(dataset_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    ratios: list[float] = []
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        relative_path = row.get("mask_filename", "")
        if not relative_path:
            continue
        mask_path = dataset_dir / relative_path
        if not mask_path.is_file():
            continue
        with Image.open(mask_path) as mask:
            ratio = float(np.count_nonzero(np.asarray(mask)) / (mask.width * mask.height))
        ratios.append(ratio)
        by_class[row.get("class", "unknown")].append(ratio)
    if not ratios:
        return {"enabled": False}
    return {
        "enabled": True,
        "foreground_ratio": _summary(ratios),
        "foreground_ratio_by_class": {name: _summary(values) for name, values in sorted(by_class.items())},
    }


def _quality_summary(dataset_dir: Path) -> dict[str, Any]:
    quality_path = dataset_dir / "quality_report.json"
    if not quality_path.is_file():
        return {"available": False}
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"available": False, "warning": "quality_report.json is invalid JSON"}
    return {
        "available": True,
        "passed": quality.get("passed"),
        "error_count": len(quality.get("errors", [])),
        "warning_count": len(quality.get("warnings", [])),
        "exact_duplicate_groups": len(quality.get("statistics", {}).get("exact_duplicate_groups", [])),
        "near_duplicate_groups": len(quality.get("statistics", {}).get("near_duplicate_groups", [])),
    }


def _confounder_warnings(by_class: dict[str, dict[str, dict[str, float | int]]]) -> list[str]:
    warnings: list[str] = []
    for field in NUMERIC_FIELDS:
        means = [float(values[field]["mean"]) for values in by_class.values() if field in values]
        if len(means) < 2:
            continue
        spread = max(means) - min(means)
        pooled_scale = pstdev(means)
        if spread > 0 and pooled_scale > 0:
            warnings.append(
                f"Class means differ for {field} (range {min(means):.4g} to {max(means):.4g}); "
                "confirm that this is an intentional experimental factor."
            )
    return warnings


def _write_figures(
    dataset_dir: Path,
    rows: list[dict[str, str]],
    numeric: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Analysis figures require matplotlib. Install it with "
            "`pip install -r requirements-analysis.txt`."
        ) from exc

    parameter_path = figures_dir / "parameter_distributions.png"
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, field in zip(axes.flat, NUMERIC_FIELDS):
        values = [float(row[field]) for row in rows if _float_or_none(row.get(field)) is not None]
        axis.hist(values, bins=min(20, max(1, len(set(values)))), color="black", edgecolor="white")
        axis.set_title(field)
        axis.set_xlabel(field)
        axis.set_ylabel("samples")
    figure.tight_layout()
    figure.savefig(parameter_path, dpi=160)
    plt.close(figure)

    montage_path = figures_dir / "class_montage.png"
    _write_montage(dataset_dir, _representative_rows(rows), montage_path, "Representative examples")
    extremes_path = figures_dir / "extreme_examples.png"
    _write_montage(dataset_dir, _extreme_rows(rows), extremes_path, "Extreme parameter examples")
    return [path.relative_to(figures_dir.parent).as_posix() for path in (parameter_path, montage_path, extremes_path)]


def _representative_rows(rows: list[dict[str, str]], per_class: int = 4) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item.get("filename", "")):
        by_class[row.get("class", "unknown")].append(row)
    for class_rows in by_class.values():
        selected.extend(class_rows[:per_class])
    return selected


def _extreme_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for field in ("shape_area", "rotation_degrees", "scale"):
        numeric_rows = [row for row in rows if _float_or_none(row.get(field)) is not None]
        if numeric_rows:
            selected[f"min_{field}"] = min(numeric_rows, key=lambda row: float(row[field]))
            selected[f"max_{field}"] = max(numeric_rows, key=lambda row: float(row[field]))
    return list(selected.values())


def _write_montage(dataset_dir: Path, rows: Iterable[dict[str, str]], path: Path, title: str) -> None:
    rows = list(rows)
    tile_size = 112
    columns = 4
    caption_height = 28
    image_rows = max(1, (len(rows) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * tile_size, 24 + image_rows * (tile_size + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill="black")
    for index, row in enumerate(rows):
        image_path = dataset_dir / row["filename"]
        with Image.open(image_path) as image:
            tile = ImageOps.contain(image.convert("RGB"), (tile_size, tile_size))
        x = (index % columns) * tile_size + (tile_size - tile.width) // 2
        y = 24 + (index // columns) * (tile_size + caption_height) + (tile_size - tile.height) // 2
        canvas.paste(tile, (x, y))
        draw.text((index % columns * tile_size + 2, y + tile_size + 2), row.get("class", "unknown"), fill="black")
    canvas.save(path, format="PNG")


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "std": pstdev(values),
    }


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Dataset Analysis Report", "", f"Samples: {report['sample_count']}", "", "## Counts", ""]
    lines.extend(["| Subset | Class | Count |", "|---|---|---:|"])
    for subset, classes in report["counts"]["by_subset_and_class"].items():
        lines.extend(f"| {subset} | {class_name} | {count} |" for class_name, count in classes.items())
    lines.extend(["", "## Potential confounders", ""])
    lines.extend(f"- {warning}" for warning in report["potential_confounders"]) if report["potential_confounders"] else lines.append("- None")
    lines.extend(["", "## Integrity and quality", "", f"- Manifest verified: {report['manifest']['verified']}"])
    quality = report["quality_report"]
    lines.append(f"- Quality report available: {quality['available']}")
    if report["figures"]:
        lines.extend(["", "## Figures", ""])
        lines.extend(f"- `{figure}`" for figure in report["figures"])
    return "\n".join(lines)
