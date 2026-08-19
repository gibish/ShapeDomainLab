from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_IMAGE_SIZES = (28, 32, 64, 128, 224, 256)
SUPPORTED_FILL_MODES = ("outline", "filled", "outline_fill")
SUPPORTED_COLOR_MODES = ("grayscale", "rgb")
SUPPORTED_IMAGE_FORMATS = ("png", "jpg", "gif")
CONFIG_SCHEMA_VERSION = "1.0"


def _validate_rgb_color(name: str, color: tuple[int, int, int]) -> None:
    if len(color) != 3:
        raise ValueError(f"{name} must contain exactly three RGB components.")
    if any(component < 0 or component > 255 for component in color):
        raise ValueError(f"{name} RGB components must be in the range 0 to 255.")


def _validate_integer_sampling_range(
    name: str,
    value: tuple[int, int],
    minimum: int,
    maximum: int | None = None,
) -> None:
    try:
        lower, upper = value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly two integer values.") from exc
    if any(not isinstance(component, int) or isinstance(component, bool) for component in (lower, upper)):
        raise ValueError(f"{name} must contain exactly two integer values.")
    if lower < minimum or upper <= lower or (maximum is not None and upper > maximum):
        bounds = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} must be an increasing range with values from {bounds}.")


@dataclass(frozen=True)
class ShapeRenderConfig:
    color_mode: str = "grayscale"
    background_range: tuple[int, int] = (150, 200)
    foreground_range: tuple[int, int] = (0, 50)
    background_color: tuple[int, int, int] = (180, 180, 180)
    foreground_color: tuple[int, int, int] = (0, 0, 0)
    line_width_range: tuple[int, int] = (1, 4)
    fill_mode: str = "outline"
    rotation_range: tuple[float, float] = (0.0, 0.0)
    scale_range: tuple[float, float] = (1.0, 1.0)
    center_offset_range: tuple[float, float] = (0.0, 0.0)
    edge_clipping: bool = False
    min_visible_ratio: float = 0.6
    noise_level: int = 0
    margin: int = 2

    def __post_init__(self) -> None:
        if self.color_mode not in SUPPORTED_COLOR_MODES:
            supported = ", ".join(SUPPORTED_COLOR_MODES)
            raise ValueError(f"Unsupported color mode '{self.color_mode}'. Supported modes: {supported}")
        _validate_rgb_color("Background color", self.background_color)
        _validate_rgb_color("Foreground color", self.foreground_color)
        _validate_integer_sampling_range("Background intensity range", self.background_range, 0, 256)
        _validate_integer_sampling_range("Foreground intensity range", self.foreground_range, 0, 256)
        _validate_integer_sampling_range("Line width range", self.line_width_range, 1)
        if self.noise_level < 0:
            raise ValueError("Noise level cannot be negative.")
        if self.margin < 0:
            raise ValueError("Margin cannot be negative.")
        if self.fill_mode not in SUPPORTED_FILL_MODES:
            supported = ", ".join(SUPPORTED_FILL_MODES)
            raise ValueError(f"Unsupported fill mode '{self.fill_mode}'. Supported modes: {supported}")
        if self.rotation_range[0] > self.rotation_range[1]:
            raise ValueError("Rotation minimum cannot be larger than rotation maximum.")
        if self.scale_range[0] <= 0 or self.scale_range[1] <= 0:
            raise ValueError("Scale range values must be positive.")
        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError("Scale minimum cannot be larger than scale maximum.")
        if self.center_offset_range[0] < 0 or self.center_offset_range[1] < 0:
            raise ValueError("Center offset range values cannot be negative.")
        if self.center_offset_range[0] > self.center_offset_range[1]:
            raise ValueError("Center offset minimum cannot be larger than center offset maximum.")
        if self.min_visible_ratio < 0.6 or self.min_visible_ratio > 1.0:
            raise ValueError("Minimum visible ratio must be in the range 0.6 to 1.0.")


@dataclass(frozen=True)
class DatasetConfig:
    config_version: str = CONFIG_SCHEMA_VERSION
    output_dir: Path = Path("datasets/shapes")
    image_size: int = 64
    classes: tuple[str, ...] = ("circle", "octagon", "rectangle", "square", "triangle")
    train_count: int = 10
    val_count: int = 10
    test_count: int = 10
    subset_totals: dict[str, int] | None = None
    file_format: str = "png"
    save_masks: bool = False
    seed: int = 12345
    render: ShapeRenderConfig = field(default_factory=ShapeRenderConfig)

    @property
    def subset_counts(self) -> dict[str, int]:
        return {
            "train": self.train_count,
            "val": self.val_count,
            "test": self.test_count,
        }

    def counts_for_subset(self, subset: str) -> dict[str, int]:
        base = self.subset_counts[subset]
        total = self.subset_totals.get(subset) if self.subset_totals else None
        if total is None:
            return {name: base for name in self.classes}
        quotient, remainder = divmod(total, len(self.classes))
        return {
            name: quotient + (index < remainder)
            for index, name in enumerate(self.classes)
        }
