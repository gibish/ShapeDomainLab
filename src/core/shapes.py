from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw

from .config import ShapeRenderConfig

ShapeRenderer = Callable[[np.random.Generator, int, ShapeRenderConfig, dict | None], Image.Image]
Point = tuple[float, float]
Color = int | tuple[int, int, int]


def _blank_image(
    rng: np.random.Generator,
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> tuple[Image.Image, ImageDraw.ImageDraw, Color, int]:
    if render_config.color_mode == "rgb":
        background: Color = render_config.background_color
        foreground: Color = render_config.foreground_color
        mode = "RGB"
    else:
        background = int(rng.integers(*render_config.background_range))
        foreground = int(rng.integers(*render_config.foreground_range))
        mode = "L"

    line_width = int(rng.integers(*render_config.line_width_range))
    if trace is not None:
        trace.update({"background": background, "foreground": foreground, "line_width": line_width})
    if render_config.noise_level > 0:
        base = np.asarray(background, dtype=np.int16)
        size = (image_size, image_size, 3) if render_config.color_mode == "rgb" else (image_size, image_size)
        noise = rng.integers(
            -render_config.noise_level,
            render_config.noise_level + 1,
            size=size,
        )
        pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
    else:
        image = Image.new(mode, (image_size, image_size), background)
    return image, ImageDraw.Draw(image), foreground, line_width


def _scale_factor(rng: np.random.Generator, render_config: ShapeRenderConfig, trace: dict | None = None) -> float:
    minimum, maximum = render_config.scale_range
    if minimum == maximum:
        value = minimum
    else:
        value = float(rng.uniform(minimum, maximum))
    if trace is not None:
        trace["scale"] = value
    return value


def _rotation_angle(rng: np.random.Generator, render_config: ShapeRenderConfig, trace: dict | None = None) -> float:
    minimum, maximum = render_config.rotation_range
    if minimum == maximum:
        degrees = minimum
    else:
        degrees = float(rng.uniform(minimum, maximum))
    if trace is not None:
        trace["rotation_degrees"] = degrees
    return float(np.deg2rad(degrees))


def _fit_points(points: list[Point], image_size: int, margin: int) -> list[Point]:
    max_radius = max(1.0, image_size / 2 - margin)
    radius = max(np.hypot(x, y) for x, y in points)
    if radius <= max_radius:
        return points
    factor = max_radius / radius
    return [(x * factor, y * factor) for x, y in points]


def _center_for_points(
    rng: np.random.Generator,
    points: list[Point],
    image_size: int,
    margin: int,
    render_config: ShapeRenderConfig,
) -> Point:
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    x_min = margin - min_x
    x_max = image_size - margin - max_x
    y_min = margin - min_y
    y_max = image_size - margin - max_y
    center = image_size / 2
    offset_min, offset_max = render_config.center_offset_range
    if offset_min == offset_max:
        offset_radius = offset_min
    else:
        offset_radius = float(rng.uniform(offset_min, offset_max))
    offset_angle = float(rng.uniform(0, 2 * np.pi))
    x = center + offset_radius * np.cos(offset_angle)
    y = center + offset_radius * np.sin(offset_angle)
    x = float(np.clip(x, x_min, x_max)) if x_min < x_max else center
    y = float(np.clip(y, y_min, y_max)) if y_min < y_max else center
    return x, y


def _center_for_clipped_points(
    rng: np.random.Generator,
    points: list[Point],
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> Point:
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    width = max_x - min_x
    height = max_y - min_y
    side = str(rng.choice(("left", "right", "top", "bottom")))
    if trace is not None:
        trace["clipping_side"] = side
    max_hidden_fraction = 1.0 - float(np.sqrt(render_config.min_visible_ratio))
    if side in {"left", "right"}:
        max_hidden = width * max_hidden_fraction
        hidden = _hidden_distance(rng, max_hidden)
        x = -hidden - min_x if side == "left" else image_size + hidden - max_x
        y = image_size / 2 - (min_y + max_y) / 2
    else:
        max_hidden = height * max_hidden_fraction
        hidden = _hidden_distance(rng, max_hidden)
        x = image_size / 2 - (min_x + max_x) / 2
        y = -hidden - min_y if side == "top" else image_size + hidden - max_y
    return float(x), float(y)


def _hidden_distance(rng: np.random.Generator, max_hidden: float) -> float:
    if max_hidden <= 1.0:
        return max(0.0, max_hidden)
    return float(rng.uniform(1.0, max_hidden))


def _rotate_points(points: list[Point], angle: float) -> list[Point]:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return [
        (
            x * cos_angle - y * sin_angle,
            x * sin_angle + y * cos_angle,
        )
        for x, y in points
    ]


def _place_points(
    rng: np.random.Generator,
    points: list[Point],
    image_size: int,
    margin: int,
    rotation: float,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> list[Point]:
    fit_margin = 0 if render_config.edge_clipping else margin
    rotated = _fit_points(_rotate_points(points, rotation), image_size, fit_margin)
    if render_config.edge_clipping:
        center_x, center_y = _center_for_clipped_points(rng, rotated, image_size, render_config, trace)
    else:
        center_x, center_y = _center_for_points(rng, rotated, image_size, margin, render_config)
    if trace is not None:
        trace.update({"center_x": center_x, "center_y": center_y})
    placed_points = [(center_x + x, center_y + y) for x, y in rotated]
    if trace is not None:
        _record_polygon_geometry(placed_points, trace)
    return placed_points


def _record_polygon_geometry(points: list[Point], trace: dict) -> None:
    """Record the rendered polygon bounds in image-coordinate space."""
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    trace.update(
        {
            "bbox_left": min(x_values),
            "bbox_top": min(y_values),
            "bbox_right": max(x_values),
            "bbox_bottom": max(y_values),
        }
    )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[Point],
    foreground: Color,
    line_width: int,
    fill_mode: str,
    trace: dict | None = None,
) -> None:
    if fill_mode in {"filled", "outline_fill"}:
        draw.polygon(points, fill=foreground)
    if fill_mode in {"outline", "outline_fill"}:
        draw.line(points + [points[0]], fill=foreground, width=line_width)
    if trace is not None:
        trace["_mask_instruction"] = ("polygon", points, fill_mode, line_width)


def _ellipse_kwargs(foreground: Color, fill_mode: str) -> dict[str, Color | None]:
    return {
        "outline": foreground if fill_mode in {"outline", "outline_fill"} else None,
        "fill": foreground if fill_mode in {"filled", "outline_fill"} else None,
    }


def render_mask_from_trace(image_size: int, trace: dict) -> Image.Image:
    """Create an exact binary foreground mask from a completed render trace."""
    try:
        kind, geometry, fill_mode, line_width = trace["_mask_instruction"]
    except KeyError as exc:
        raise ValueError("Render trace does not contain a mask instruction.") from exc

    mask = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "polygon":
        points = geometry
        if fill_mode in {"filled", "outline_fill"}:
            draw.polygon(points, fill=255)
        if fill_mode in {"outline", "outline_fill"}:
            draw.line(points + [points[0]], fill=255, width=line_width)
    elif kind == "ellipse":
        draw.ellipse(
            geometry,
            outline=255 if fill_mode in {"outline", "outline_fill"} else None,
            fill=255 if fill_mode in {"filled", "outline_fill"} else None,
            width=line_width,
        )
    else:
        raise ValueError(f"Unsupported mask instruction: {kind}")
    return mask


def render_circle(
    rng: np.random.Generator,
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> Image.Image:
    image, draw, foreground, line_width = _blank_image(rng, image_size, render_config, trace)
    margin = render_config.margin
    radius_min = max(4, image_size // 10)
    radius_max = max(radius_min + 1, image_size // 4)
    radius = max(1, int(rng.integers(radius_min, radius_max) * _scale_factor(rng, render_config, trace)))
    radius = min(radius, max(1, image_size // 2 - margin))
    if trace is not None:
        trace.update({"radius": radius, "rotation_degrees": 0.0})
    if render_config.edge_clipping:
        points = [(-radius, -radius), (radius, radius)]
        x, y = _center_for_clipped_points(rng, points, image_size, render_config, trace)
    else:
        center = image_size / 2
        offset_min, offset_max = render_config.center_offset_range
        offset_radius = offset_min if offset_min == offset_max else float(rng.uniform(offset_min, offset_max))
        offset_angle = float(rng.uniform(0, 2 * np.pi))
        x = center + offset_radius * np.cos(offset_angle)
        y = center + offset_radius * np.sin(offset_angle)
        x = int(np.clip(x, margin + radius, image_size - margin - radius))
        y = int(np.clip(y, margin + radius, image_size - margin - radius))
    if trace is not None:
        trace.update(
            {
                "center_x": x,
                "center_y": y,
                "shape_width": 2 * radius,
                "shape_height": 2 * radius,
                "shape_area": float(np.pi * radius**2),
                "bbox_left": x - radius,
                "bbox_top": y - radius,
                "bbox_right": x + radius,
                "bbox_bottom": y + radius,
            }
        )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        **_ellipse_kwargs(foreground, render_config.fill_mode),
        width=line_width,
    )
    if trace is not None:
        trace["_mask_instruction"] = (
            "ellipse",
            (x - radius, y - radius, x + radius, y + radius),
            render_config.fill_mode,
            line_width,
        )
    return image


def render_square(
    rng: np.random.Generator,
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> Image.Image:
    image, draw, foreground, line_width = _blank_image(rng, image_size, render_config, trace)
    margin = render_config.margin
    side_min = max(8, image_size // 5)
    side_max = max(side_min + 1, image_size // 2)
    side = max(2, int(rng.integers(side_min, side_max) * _scale_factor(rng, render_config, trace)))
    if trace is not None:
        trace.update({"shape_width": side, "shape_height": side, "shape_area": side * side})
    half = side / 2
    points = [(-half, -half), (half, -half), (half, half), (-half, half)]
    placed_points = _place_points(rng, points, image_size, margin, _rotation_angle(rng, render_config, trace), render_config, trace)
    _draw_polygon(draw, placed_points, foreground, line_width, render_config.fill_mode, trace)
    return image


def render_rectangle(
    rng: np.random.Generator,
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> Image.Image:
    image, draw, foreground, line_width = _blank_image(rng, image_size, render_config, trace)
    margin = render_config.margin
    side_min = max(8, image_size // 6)
    side_max = max(side_min + 1, image_size // 2)
    scale = _scale_factor(rng, render_config, trace)
    width = max(2, int(rng.integers(side_min, side_max) * scale))
    height = max(2, int(rng.integers(side_min, side_max) * scale))
    while abs(width - height) < max(4, image_size // 12):
        height = max(2, int(rng.integers(side_min, side_max) * scale))
    if trace is not None:
        trace.update({"shape_width": width, "shape_height": height, "shape_area": width * height})
    half_width = width / 2
    half_height = height / 2
    points = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ]
    placed_points = _place_points(rng, points, image_size, margin, _rotation_angle(rng, render_config, trace), render_config, trace)
    _draw_polygon(draw, placed_points, foreground, line_width, render_config.fill_mode, trace)
    return image


def render_triangle(
    rng: np.random.Generator,
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> Image.Image:
    image, draw, foreground, line_width = _blank_image(rng, image_size, render_config, trace)
    margin = render_config.margin
    base_min = max(10, image_size // 5)
    base_max = max(base_min + 1, image_size // 2)
    scale = _scale_factor(rng, render_config, trace)
    base = max(2, int(rng.integers(base_min, base_max) * scale))
    height = max(2, int(rng.integers(base_min, base_max) * scale))
    apex_offset = float(rng.uniform(0, base))
    if trace is not None:
        trace.update({"shape_width": base, "shape_height": height, "shape_area": base * height / 2})
    points = [
        (-base / 2, height / 2),
        (base / 2, height / 2),
        (apex_offset - base / 2, -height / 2),
    ]
    placed_points = _place_points(rng, points, image_size, margin, _rotation_angle(rng, render_config, trace), render_config, trace)
    _draw_polygon(draw, placed_points, foreground, line_width, render_config.fill_mode, trace)
    return image


def render_octagon(
    rng: np.random.Generator,
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> Image.Image:
    image, draw, foreground, line_width = _blank_image(rng, image_size, render_config, trace)
    margin = render_config.margin
    radius_min = max(6, image_size // 8)
    radius_max = max(radius_min + 1, image_size // 3)
    radius = max(2, int(rng.integers(radius_min, radius_max) * _scale_factor(rng, render_config, trace)))
    if trace is not None:
        trace.update(
            {
                "radius": radius,
                "shape_width": 2 * radius,
                "shape_height": 2 * radius,
                "shape_area": float(2 * np.sqrt(2) * radius**2),
            }
        )
    rotation = _rotation_angle(rng, render_config, trace)
    points = [
        (
            radius * np.cos(index * np.pi / 4),
            radius * np.sin(index * np.pi / 4),
        )
        for index in range(8)
    ]
    placed_points = _place_points(rng, points, image_size, margin, rotation, render_config, trace)
    _draw_polygon(draw, placed_points, foreground, line_width, render_config.fill_mode, trace)
    return image


RENDERERS: dict[str, ShapeRenderer] = {
    "circle": render_circle,
    "octagon": render_octagon,
    "rectangle": render_rectangle,
    "square": render_square,
    "triangle": render_triangle,
}


def render_shape(
    class_name: str,
    rng: np.random.Generator,
    image_size: int,
    render_config: ShapeRenderConfig,
    trace: dict | None = None,
) -> Image.Image:
    try:
        renderer = RENDERERS[class_name]
    except KeyError as exc:
        supported = ", ".join(sorted(RENDERERS))
        raise ValueError(f"Unsupported shape class '{class_name}'. Supported: {supported}") from exc
    return renderer(rng, image_size, render_config, trace)
