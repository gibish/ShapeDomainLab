from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from .config import DatasetConfig
from .generator import IMAGE_EXTENSIONS, normalize_file_format
from .manifest import DATASET_MANIFEST_FILENAME
from .preview import generate_preview_images


@dataclass(frozen=True)
class DatasetSummary:
    output_dir: Path
    exists: bool
    subsets: dict[str, dict[str, int]] = field(default_factory=dict)
    total_images: int = 0
    service_files: dict[str, bool] = field(default_factory=dict)
    image_formats: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetEstimate:
    total_images: int
    sample_count: int
    estimated_image_bytes: int
    estimated_service_bytes: int

    @property
    def estimated_total_bytes(self) -> int:
        return self.estimated_image_bytes + self.estimated_service_bytes


def dataset_summary(output_dir: Path | str) -> DatasetSummary:
    root = Path(output_dir)
    if not root.exists():
        return DatasetSummary(output_dir=root, exists=False)

    subsets: dict[str, dict[str, int]] = {}
    image_formats: dict[str, int] = {}
    total_images = 0
    for subset in ("train", "val", "test"):
        subset_path = root / subset
        if not subset_path.is_dir():
            continue
        class_counts: dict[str, int] = {}
        for class_dir in sorted(path for path in subset_path.iterdir() if path.is_dir()):
            images = [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            class_counts[class_dir.name] = len(images)
            total_images += len(images)
            for image_path in images:
                suffix = image_path.suffix.lower().lstrip(".")
                image_formats[suffix] = image_formats.get(suffix, 0) + 1
        subsets[subset] = class_counts

    service_files = {
        filename: (root / filename).is_file()
        for filename in ("metadata.csv", "class_mapping.json", "generation_config.json", "log.txt", "README.md", DATASET_MANIFEST_FILENAME)
    }
    return DatasetSummary(
        output_dir=root,
        exists=True,
        subsets=subsets,
        total_images=total_images,
        service_files=service_files,
        image_formats=image_formats,
    )


def estimate_dataset_size(config: DatasetConfig, sample_limit: int = 8) -> DatasetEstimate:
    total_images = sum(config.subset_totals.values()) if config.subset_totals else sum(config.subset_counts.values()) * len(config.classes)
    if total_images == 0:
        return DatasetEstimate(0, 0, 0, 0)

    previews = generate_preview_images(config, min(sample_limit, total_images))
    file_format = normalize_file_format(config.file_format)
    image_bytes = [_encoded_size(preview.image, file_format) for preview in previews]
    average_image_bytes = int(sum(image_bytes) / len(image_bytes)) if image_bytes else 0
    estimated_image_bytes = average_image_bytes * total_images
    estimated_service_bytes = 50_000 + total_images * 160
    return DatasetEstimate(
        total_images=total_images,
        sample_count=len(previews),
        estimated_image_bytes=estimated_image_bytes,
        estimated_service_bytes=estimated_service_bytes,
    )


def _encoded_size(image, file_format: str) -> int:
    buffer = BytesIO()
    pillow_format = "JPEG" if file_format == "jpg" else file_format.upper()
    image.save(buffer, format=pillow_format)
    return len(buffer.getvalue())
