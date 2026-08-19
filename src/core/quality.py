import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

from .config import DatasetConfig
from .metadata import class_mapping
from .manifest import DATASET_MANIFEST_FILENAME, verify_dataset_manifest
from .paths import class_dir, mask_dir


PIL_FORMATS = {"png": "PNG", "jpg": "JPEG", "gif": "GIF"}
QUALITY_REPORT_JSON = "quality_report.json"
QUALITY_REPORT_MARKDOWN = "quality_report.md"
SMALL_FOREGROUND_RATIO = 0.005
LARGE_FOREGROUND_RATIO = 0.90


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)


def validate_dataset(output_dir: Path, config: DatasetConfig, write_report: bool = True) -> QualityReport:
    """Validate dataset artifacts and calculate reproducible quality statistics."""
    errors: list[str] = []
    warnings: list[str] = []
    output_dir = Path(output_dir)
    file_format = config.file_format.lower().lstrip(".")

    if not output_dir.exists():
        return QualityReport(False, [f"Dataset directory does not exist: {output_dir}"], warnings)

    image_paths: list[Path] = []
    for subset, class_counts in ((subset, config.counts_for_subset(subset)) for subset in config.subset_counts):
        subset_path = output_dir / subset
        if not subset_path.is_dir():
            errors.append(f"Missing subset directory: {subset}")
            continue

        for class_name in config.classes:
            current_class_dir = class_dir(output_dir, subset, class_name)
            if not current_class_dir.is_dir():
                errors.append(f"Missing class directory: {subset}/{class_name}")
                continue

            expected_count = class_counts[class_name]
            files = sorted(current_class_dir.glob(f"*.{file_format}"))
            image_paths.extend(files)
            if len(files) != expected_count:
                errors.append(f"Expected {expected_count} images in {subset}/{class_name}, found {len(files)}")
            for image_path in files:
                _validate_image(image_path, config, errors)

    _validate_service_files(output_dir, errors)
    manifest_errors, manifest_statistics = verify_dataset_manifest(output_dir)
    errors.extend(manifest_errors)
    _validate_class_mapping(output_dir, config, errors)
    metadata_rows = _validate_metadata(output_dir, config, image_paths, errors)
    if config.save_masks:
        _validate_masks(output_dir, config, errors)

    statistics = _analyze_dataset(output_dir, config, image_paths, metadata_rows, errors, warnings)
    statistics["manifest"] = manifest_statistics
    report = QualityReport(not errors, errors, warnings, statistics)
    if write_report:
        write_quality_reports(output_dir, report)
    return report


def write_quality_reports(output_dir: Path, report: QualityReport) -> None:
    """Write machine-readable and human-readable quality reports for a dataset."""
    payload = {
        "passed": report.passed,
        "errors": report.errors,
        "warnings": report.warnings,
        "statistics": report.statistics,
    }
    (output_dir / QUALITY_REPORT_JSON).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# Dataset Quality Report", "", f"Status: {'passed' if report.passed else 'failed'}", ""]
    _append_markdown_list(lines, "Errors", report.errors)
    _append_markdown_list(lines, "Warnings", report.warnings)
    lines.extend(["## Statistics", "", "```json", json.dumps(report.statistics, indent=2), "```", ""])
    (output_dir / QUALITY_REPORT_MARKDOWN).write_text("\n".join(lines), encoding="utf-8")


def _append_markdown_list(lines: list[str], title: str, entries: list[str]) -> None:
    lines.extend([f"## {title}", ""])
    if entries:
        lines.extend(f"- {entry}" for entry in entries)
    else:
        lines.append("- None")
    lines.append("")


def _validate_image(image_path: Path, config: DatasetConfig, errors: list[str]) -> None:
    try:
        with Image.open(image_path) as image:
            expected_size = (config.image_size, config.image_size)
            if image.size != expected_size:
                errors.append(f"Invalid image size for {image_path}: expected {expected_size}, found {image.size}")
            file_format = config.file_format.strip().lower().lstrip(".")
            expected_format = PIL_FORMATS[file_format]
            if image.format != expected_format:
                errors.append(f"Invalid image format for {image_path}: expected {expected_format}, found {image.format}")
            expected_mode = "P" if file_format == "gif" else ("RGB" if config.render.color_mode == "rgb" else "L")
            if image.mode != expected_mode:
                errors.append(f"Invalid image mode for {image_path}: expected {expected_mode}, found {image.mode}")
    except OSError as exc:
        errors.append(f"Cannot open image {image_path}: {exc}")


def _validate_masks(output_dir: Path, config: DatasetConfig, errors: list[str]) -> None:
    for subset, class_counts in ((subset, config.counts_for_subset(subset)) for subset in config.subset_counts):
        for class_name in config.classes:
            current_mask_dir = mask_dir(output_dir, subset, class_name)
            if not current_mask_dir.is_dir():
                errors.append(f"Missing mask directory: masks/{subset}/{class_name}")
                continue
            masks = sorted(current_mask_dir.glob("*.png"))
            expected_count = class_counts[class_name]
            if len(masks) != expected_count:
                errors.append(f"Expected {expected_count} masks in masks/{subset}/{class_name}, found {len(masks)}")
            for mask_path in masks:
                _validate_mask(mask_path, config, errors)


def _validate_mask(mask_path: Path, config: DatasetConfig, errors: list[str]) -> None:
    try:
        with Image.open(mask_path) as mask:
            if mask.format != "PNG":
                errors.append(f"Invalid mask format for {mask_path}: expected PNG, found {mask.format}")
            if mask.mode != "L":
                errors.append(f"Invalid mask mode for {mask_path}: expected L, found {mask.mode}")
            if mask.size != (config.image_size, config.image_size):
                errors.append(f"Invalid mask size for {mask_path}: expected {(config.image_size, config.image_size)}, found {mask.size}")
            values = set(mask.getdata())
            if not values <= {0, 255}:
                errors.append(f"Mask is not binary: {mask_path}")
            if 255 not in values:
                errors.append(f"Mask contains no foreground: {mask_path}")
    except OSError as exc:
        errors.append(f"Cannot open mask {mask_path}: {exc}")


def _validate_service_files(output_dir: Path, errors: list[str]) -> None:
    for filename in ("metadata.csv", "class_mapping.json", "generation_config.json", "log.txt", "README.md", DATASET_MANIFEST_FILENAME):
        if not (output_dir / filename).is_file():
            errors.append(f"Missing service file: {filename}")


def _validate_class_mapping(output_dir: Path, config: DatasetConfig, errors: list[str]) -> None:
    mapping_path = output_dir / "class_mapping.json"
    if not mapping_path.is_file():
        return
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid class_mapping.json: {exc}")
        return
    expected_mapping = class_mapping(config.classes)
    if mapping != expected_mapping:
        errors.append(f"Invalid class mapping: expected {expected_mapping}, found {mapping}")


def _validate_metadata(
    output_dir: Path, config: DatasetConfig, image_paths: list[Path], errors: list[str]
) -> list[dict[str, str]]:
    metadata_path = output_dir / "metadata.csv"
    if not metadata_path.is_file():
        return []
    try:
        with metadata_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except csv.Error as exc:
        errors.append(f"Invalid metadata.csv: {exc}")
        return []

    image_relative_paths = {path.relative_to(output_dir).as_posix() for path in image_paths}
    metadata_relative_paths = {row.get("filename", "") for row in rows}
    if len(rows) != len(image_paths):
        errors.append(f"Metadata row count mismatch: expected {len(image_paths)}, found {len(rows)}")
    missing_files = sorted(metadata_relative_paths - image_relative_paths)
    if missing_files:
        errors.append(f"Metadata references missing files: {', '.join(missing_files[:5])}")
    missing_metadata = sorted(image_relative_paths - metadata_relative_paths)
    if missing_metadata:
        errors.append(f"Images missing metadata rows: {', '.join(missing_metadata[:5])}")

    for row in rows:
        filename = row.get("filename", "")
        if row.get("subset") not in config.subset_counts:
            errors.append(f"Invalid subset in metadata for {filename}: {row.get('subset')}")
        if row.get("class") not in config.classes:
            errors.append(f"Invalid class in metadata for {filename}: {row.get('class')}")
        if row.get("status") != "ok":
            errors.append(f"Invalid status in metadata for {filename}: {row.get('status')}")
        mask_filename = row.get("mask_filename", "")
        if config.save_masks:
            if not mask_filename:
                errors.append(f"Missing mask filename in metadata for {filename}")
            elif not (output_dir / mask_filename).is_file():
                errors.append(f"Metadata references missing mask: {mask_filename}")
        elif mask_filename:
            errors.append(f"Unexpected mask filename in metadata for {filename}")
    return rows


def _analyze_dataset(
    output_dir: Path,
    config: DatasetConfig,
    image_paths: list[Path],
    metadata_rows: list[dict[str, str]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect duplicate, geometry, clipping, and optional mask-occupancy statistics."""
    hashes: dict[str, list[Path]] = defaultdict(list)
    perceptual_hashes: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_paths:
        try:
            hashes[_file_hash(image_path)].append(image_path)
            perceptual_hashes[_average_hash(image_path)].append(image_path)
        except OSError as exc:
            warnings.append(f"Skipped duplicate analysis for unreadable image {image_path}: {exc}")

    exact_duplicates = _duplicate_groups(hashes, output_dir)
    near_duplicates = _duplicate_groups(perceptual_hashes, output_dir)
    cross_subset_duplicates = [group for group in exact_duplicates if len({_subset_from_path(path) for path in group}) > 1]
    if exact_duplicates:
        warnings.append(f"Detected {len(exact_duplicates)} exact duplicate image group(s).")
    if near_duplicates:
        warnings.append(f"Detected {len(near_duplicates)} near-duplicate image group(s) by average hash.")
    if cross_subset_duplicates:
        errors.append(f"Detected {len(cross_subset_duplicates)} exact duplicate group(s) across dataset subsets.")

    geometry = _geometry_statistics(metadata_rows, config, errors, warnings)
    mask_statistics = _mask_statistics(output_dir, metadata_rows, config, errors, warnings)
    return {
        "total_images": len(image_paths),
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_groups": near_duplicates,
        "cross_subset_exact_duplicate_groups": cross_subset_duplicates,
        "geometry": geometry,
        "masks": mask_statistics,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((8, 8)), dtype=np.float32)
    bits = pixels >= pixels.mean()
    return "".join("1" if bit else "0" for bit in bits.flat)


def _duplicate_groups(groups: dict[str, list[Path]], output_dir: Path) -> list[list[str]]:
    return [
        sorted(path.relative_to(output_dir).as_posix() for path in paths)
        for paths in groups.values()
        if len(paths) > 1
    ]


def _subset_from_path(relative_path: str) -> str:
    return relative_path.split("/", maxsplit=1)[0]


def _geometry_statistics(
    rows: list[dict[str, str]], config: DatasetConfig, errors: list[str], warnings: list[str]
) -> dict[str, Any]:
    values_by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    clipped = 0
    for row in rows:
        class_name = row.get("class", "unknown")
        for field in ("scale", "rotation_degrees", "center_x", "center_y", "line_width", "shape_area"):
            value = _float_or_none(row.get(field, ""))
            if value is not None:
                values_by_class[class_name][field].append(value)
        area = _float_or_none(row.get("shape_area", ""))
        if area is not None:
            area_ratio = area / (config.image_size**2)
            if area_ratio < SMALL_FOREGROUND_RATIO:
                warnings.append(f"Nominal shape area is very small for {row.get('filename', '')}: {area_ratio:.4f}.")
            if area_ratio > LARGE_FOREGROUND_RATIO:
                warnings.append(f"Nominal shape area is very large for {row.get('filename', '')}: {area_ratio:.4f}.")
        bounds = tuple(_float_or_none(row.get(field, "")) for field in ("bbox_left", "bbox_top", "bbox_right", "bbox_bottom"))
        if all(value is not None for value in bounds):
            left, top, right, bottom = bounds
            outside = left < 0 or top < 0 or right > config.image_size or bottom > config.image_size
            if outside and row.get("edge_clipping") != "True":
                errors.append(f"Shape bounds are outside the image without edge clipping: {row.get('filename', '')}")
        if row.get("edge_clipping") == "True":
            clipped += 1

    per_class: dict[str, dict[str, dict[str, float]]] = {}
    for class_name, field_values in values_by_class.items():
        per_class[class_name] = {
            field: {"min": min(values), "max": max(values), "mean": mean(values)}
            for field, values in field_values.items()
            if values
        }
    return {"clipped_images": clipped, "per_class": per_class}


def _mask_statistics(
    output_dir: Path,
    rows: list[dict[str, str]],
    config: DatasetConfig,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if not config.save_masks:
        return {"enabled": False}
    ratios: list[float] = []
    per_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        mask_filename = row.get("mask_filename", "")
        if not mask_filename:
            continue
        try:
            with Image.open(output_dir / mask_filename) as mask:
                ratio = float(np.count_nonzero(np.asarray(mask)) / mask.size[0] / mask.size[1])
        except OSError:
            continue
        ratios.append(ratio)
        per_class[row.get("class", "unknown")].append(ratio)
        if ratio == 0:
            errors.append(f"Mask has no visible foreground: {mask_filename}")
        elif ratio < SMALL_FOREGROUND_RATIO:
            warnings.append(f"Visible foreground is very small in mask: {mask_filename} ({ratio:.4f}).")
        elif ratio > LARGE_FOREGROUND_RATIO:
            warnings.append(f"Visible foreground is very large in mask: {mask_filename} ({ratio:.4f}).")
    return {
        "enabled": True,
        "foreground_ratio": _summary(ratios),
        "foreground_ratio_by_class": {class_name: _summary(values) for class_name, values in per_class.items()},
    }


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {"count": len(values), "min": min(values), "max": max(values), "mean": mean(values)}


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
