from .config import DatasetConfig, SUPPORTED_IMAGE_FORMATS, SUPPORTED_IMAGE_SIZES
from .shapes import RENDERERS


def validate_config(config: DatasetConfig) -> list[str]:
    errors: list[str] = []

    if not isinstance(config.config_version, str) or not config.config_version.strip():
        errors.append("Configuration version must be a non-empty string.")

    if config.image_size not in SUPPORTED_IMAGE_SIZES:
        supported = ", ".join(str(size) for size in SUPPORTED_IMAGE_SIZES)
        errors.append(f"Unsupported image size {config.image_size}. Supported sizes: {supported}")

    if config.train_count < 1:
        errors.append("Train count must be at least 1.")
    if config.test_count < 1:
        errors.append("Test count must be at least 1.")
    if config.val_count < 1:
        errors.append("Validation count must be at least 1.")
    if config.subset_totals is not None:
        if not isinstance(config.subset_totals, dict):
            errors.append("Subset totals must be an object with train, val, and test values.")
        else:
            expected_subsets = set(config.subset_counts)
            supplied_subsets = set(config.subset_totals)
            missing_subsets = sorted(expected_subsets - supplied_subsets)
            if missing_subsets:
                errors.append(f"Missing subset total(s): {', '.join(missing_subsets)}")
            unsupported_subsets = sorted(supplied_subsets - expected_subsets)
            for subset in unsupported_subsets:
                errors.append(f"Unsupported subset total: {subset}")
            for subset in expected_subsets & supplied_subsets:
                total = config.subset_totals[subset]
                if not isinstance(total, int) or isinstance(total, bool):
                    errors.append(f"Total image count for {subset} must be an integer.")
                elif total < 1:
                    errors.append(f"Total image count for {subset} must be at least 1.")

    if config.seed < 0:
        errors.append("Seed cannot be negative.")

    if not isinstance(config.save_masks, bool):
        errors.append("Save masks must be a boolean value.")

    normalized_format = config.file_format.strip().lower().lstrip(".")
    if normalized_format not in SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(SUPPORTED_IMAGE_FORMATS)
        errors.append(f"Unsupported image format '{config.file_format}'. Supported formats: {supported}")

    if not config.classes:
        errors.append("Select at least one shape class.")
    else:
        unsupported = sorted(set(config.classes) - set(RENDERERS))
        if unsupported:
            supported = ", ".join(sorted(RENDERERS))
            errors.append(
                f"Unsupported shape class(es): {', '.join(unsupported)}. Supported classes: {supported}"
            )
        if len(set(config.classes)) != len(config.classes):
            errors.append("Shape classes must be unique.")

    max_center_offset = config.image_size / 2
    if config.render.center_offset_range[1] > max_center_offset:
        errors.append(
            "Center offset maximum cannot be larger than half the image size "
            f"({max_center_offset:g} pixels)."
        )

    if config.render.margin > config.image_size / 2:
        errors.append("Margin cannot be larger than half the image size.")

    return errors


def require_valid_config(config: DatasetConfig) -> None:
    errors = validate_config(config)
    if errors:
        raise ValueError("\n".join(errors))
