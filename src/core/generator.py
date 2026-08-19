from pathlib import Path
import shutil
import hashlib
import io
import sys
from typing import Callable

import numpy as np
import PIL

from src.__version__ import __version__
from .config import DatasetConfig, SUPPORTED_IMAGE_FORMATS
from .metadata import METADATA_SCHEMA_VERSION, MetadataRow, class_mapping, write_dataset_service_files
from .manifest import DATASET_MANIFEST_FILENAME
from .paths import class_dir, mask_dir, project_root, resolve_output_dir
from .shapes import render_mask_from_trace, render_shape
from .validation import require_valid_config

ProgressCallback = Callable[[int, int, str], None]
OverwriteCallback = Callable[[Path], bool]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SERVICE_FILENAMES = (
    "metadata.csv",
    "class_mapping.json",
    "generation_config.json",
    "log.txt",
    "README.md",
    DATASET_MANIFEST_FILENAME,
)
MAX_DUPLICATE_ATTEMPTS = 100
PIL_SAVE_FORMATS = {"png": "PNG", "jpg": "JPEG", "gif": "GIF"}


def prepare_dataset_dirs(config: DatasetConfig, confirm_overwrite: OverwriteCallback | None = None) -> Path:
    output_dir = resolve_output_dir(config.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            try:
                validate_generated_dataset_dir(output_dir)
            except ValueError as exc:
                raise ValueError(
                    f"Output directory is not empty and is not a generated dataset: {output_dir}. "
                    "No files were removed."
                ) from exc
            if confirm_overwrite is None or not confirm_overwrite(output_dir):
                raise ValueError(f"Generation cancelled; existing dataset was not changed: {output_dir}")
            clear_dataset_dir(output_dir)
    for subset in config.subset_counts:
        for class_name in config.classes:
            class_dir(output_dir, subset, class_name).mkdir(parents=True, exist_ok=True)
            if config.save_masks:
                mask_dir(output_dir, subset, class_name).mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_generated_dataset_dir(output_dir: Path | str) -> Path:
    resolved_output_dir = resolve_output_dir(output_dir).resolve()
    if resolved_output_dir == project_root().resolve():
        raise ValueError(f"Refusing to clear the project root: {resolved_output_dir}")
    if resolved_output_dir == Path(resolved_output_dir.anchor):
        raise ValueError(f"Refusing to clear a filesystem root: {resolved_output_dir}")
    if not resolved_output_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {resolved_output_dir}")
    if not resolved_output_dir.is_dir():
        raise ValueError(f"Dataset path is not a directory: {resolved_output_dir}")

    missing_subsets = [subset for subset in ("train", "val", "test") if not (resolved_output_dir / subset).is_dir()]
    if missing_subsets:
        missing = ", ".join(missing_subsets)
        raise ValueError(f"Directory is not a generated dataset. Missing subset folder(s): {missing}")

    image_files = [
        path
        for subset in ("train", "val", "test")
        for path in (resolved_output_dir / subset).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        raise ValueError("Directory is not a generated dataset. No image files found in train/val/test folders.")

    return resolved_output_dir


def clear_dataset_dir(output_dir: Path | str) -> Path:
    resolved_output_dir = validate_generated_dataset_dir(output_dir)
    for child in resolved_output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return resolved_output_dir


def _cleanup_failed_generation(output_dir: Path, config: DatasetConfig) -> None:
    """Remove only artifacts created by an interrupted generation run."""
    for subset in config.subset_counts:
        subset_path = output_dir / subset
        if subset_path.is_dir():
            shutil.rmtree(subset_path)
    masks_path = output_dir / "masks"
    if masks_path.is_dir():
        shutil.rmtree(masks_path)
    for filename in SERVICE_FILENAMES:
        service_path = output_dir / filename
        if service_path.is_file():
            service_path.unlink()


def normalize_file_format(file_format: str) -> str:
    normalized = file_format.strip().lower().lstrip(".")
    if normalized not in SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(SUPPORTED_IMAGE_FORMATS)
        raise ValueError(f"Unsupported image format '{file_format}'. Supported formats: {supported}")
    return normalized


def sample_seed(global_seed: int, subset: str, class_name: str, index: int, attempt: int = 0) -> int:
    """Return a stable, independent seed for one dataset sample attempt."""
    key = f"{global_seed}\0{subset}\0{class_name}\0{index}".encode("utf-8")
    if attempt:
        key += f"\0{attempt}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _encoded_image_bytes(image, file_format: str) -> bytes:
    """Encode an image once so duplicate detection matches the saved artifact."""
    buffer = io.BytesIO()
    image.save(buffer, format=PIL_SAVE_FORMATS[file_format])
    return buffer.getvalue()


def _render_unique_image(
    config: DatasetConfig,
    subset: str,
    class_name: str,
    index: int,
    file_format: str,
    used_hashes: set[str],
) -> tuple[object, dict[str, object], int, bytes]:
    """Render a sample whose encoded bytes are unique within the dataset."""
    for attempt in range(MAX_DUPLICATE_ATTEMPTS):
        trace: dict[str, object] = {}
        current_seed = sample_seed(config.seed, subset, class_name, index, attempt)
        image = render_shape(class_name, np.random.default_rng(current_seed), config.image_size, config.render, trace)
        image_bytes = _encoded_image_bytes(image, file_format)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash not in used_hashes:
            used_hashes.add(image_hash)
            return image, trace, current_seed, image_bytes
    raise ValueError(
        "Unable to generate a unique image after "
        f"{MAX_DUPLICATE_ATTEMPTS} attempts for {subset}/{class_name}/{index}. "
        "Increase the rendering variability or reduce the requested dataset size."
    )


def validate_dataset_config(config: DatasetConfig) -> None:
    require_valid_config(config)


def generate_dataset(
    config: DatasetConfig,
    progress_callback: ProgressCallback | None = None,
    confirm_overwrite: OverwriteCallback | None = None,
) -> Path:
    file_format = normalize_file_format(config.file_format)
    validate_dataset_config(config)
    output_dir = prepare_dataset_dirs(config, confirm_overwrite)
    mapping = class_mapping(config.classes)
    metadata_rows: list[MetadataRow] = []
    subset_class_counts = {subset: config.counts_for_subset(subset) for subset in config.subset_counts}
    total_images = sum(sum(counts.values()) for counts in subset_class_counts.values())
    completed_images = 0
    used_hashes: set[str] = set()

    try:
        for subset, class_counts in subset_class_counts.items():
            for class_name in config.classes:
                target_dir = class_dir(output_dir, subset, class_name)
                count = class_counts[class_name]
                for index in range(count):
                    image, trace, current_seed, image_bytes = _render_unique_image(
                        config, subset, class_name, index, file_format, used_hashes
                    )
                    filename = f"ssc_{subset}_{class_name}_{index:06d}.{file_format}"
                    image_path = target_dir / filename
                    image_path.write_bytes(image_bytes)
                    mask_filename = ""
                    if config.save_masks:
                        mask_filename = f"ssc_{subset}_{class_name}_{index:06d}.png"
                        mask_path = mask_dir(output_dir, subset, class_name) / mask_filename
                        render_mask_from_trace(config.image_size, trace).save(mask_path, format="PNG")
                    metadata_rows.append(
                        {
                            "filename": image_path.relative_to(output_dir).as_posix(),
                            "mask_filename": (
                                (mask_path.relative_to(output_dir).as_posix()) if config.save_masks else ""
                            ),
                            "metadata_schema_version": METADATA_SCHEMA_VERSION,
                            "application_version": __version__,
                            "python_version": sys.version.split()[0],
                            "numpy_version": np.__version__,
                            "pillow_version": PIL.__version__,
                            "subset": subset,
                            "class": class_name,
                            "class_id": mapping[class_name],
                            "image_size": config.image_size,
                            "shape_type": class_name,
                            "seed": current_seed,
                            "status": "ok",
                            "background": trace.get("background", ""),
                            "foreground": trace.get("foreground", ""),
                            "line_width": trace.get("line_width", ""),
                            "scale": trace.get("scale", ""),
                            "rotation_degrees": trace.get("rotation_degrees", ""),
                            "center_x": trace.get("center_x", ""),
                            "center_y": trace.get("center_y", ""),
                            "radius": trace.get("radius", ""),
                            "shape_width": trace.get("shape_width", ""),
                            "shape_height": trace.get("shape_height", ""),
                            "shape_area": trace.get("shape_area", ""),
                            "bbox_left": trace.get("bbox_left", ""),
                            "bbox_top": trace.get("bbox_top", ""),
                            "bbox_right": trace.get("bbox_right", ""),
                            "bbox_bottom": trace.get("bbox_bottom", ""),
                            "color_mode": config.render.color_mode,
                            "fill_mode": config.render.fill_mode,
                            "noise_level": config.render.noise_level,
                            "margin": config.render.margin,
                            "edge_clipping": config.render.edge_clipping,
                            "min_visible_ratio": config.render.min_visible_ratio,
                            "clipping_side": trace.get("clipping_side", ""),
                        }
                    )
                    completed_images += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_images,
                            total_images,
                            f"Generated {subset}/{class_name}/{filename}",
                        )

        write_dataset_service_files(output_dir, config, metadata_rows)
    except Exception:
        _cleanup_failed_generation(output_dir, config)
        raise

    return output_dir
