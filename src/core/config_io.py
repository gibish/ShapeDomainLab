import json
from pathlib import Path
from typing import Any

from .config import CONFIG_SCHEMA_VERSION, DatasetConfig, ShapeRenderConfig
from .metadata import config_to_dict
from .validation import require_valid_config


def load_config_from_json(path: Path | str) -> DatasetConfig:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config file: {config_path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    _require_config_format(data)

    render_data = data.get("render", {})
    if not isinstance(render_data, dict):
        raise ValueError("Config field 'render' must be an object.")

    render_kwargs = _filter_kwargs(render_data, ShapeRenderConfig)
    for key in (
        "background_range",
        "foreground_range",
        "background_color",
        "foreground_color",
        "line_width_range",
        "rotation_range",
        "scale_range",
        "center_offset_range",
    ):
        if key in render_kwargs:
            render_kwargs[key] = tuple(render_kwargs[key])
    render = ShapeRenderConfig(**render_kwargs)
    config_data = {key: value for key, value in data.items() if key != "render"}
    if "config_version" not in config_data:
        config_data["config_version"] = CONFIG_SCHEMA_VERSION
    if "val_count" not in config_data:
        config_data["val_count"] = config_data.get("test_count", DatasetConfig().val_count)
    if "output_dir" in config_data:
        config_data["output_dir"] = Path(config_data["output_dir"])
    if "classes" in config_data:
        config_data["classes"] = tuple(config_data["classes"])

    config = DatasetConfig(**_filter_kwargs(config_data, DatasetConfig), render=render)
    require_valid_config(config)
    return config


def save_config_template(path: Path | str, config: DatasetConfig | None = None) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    selected_config = config if config is not None else DatasetConfig()
    require_valid_config(selected_config)
    with target_path.open("w", encoding="utf-8") as file:
        json.dump(config_to_dict(selected_config), file, indent=2)
        file.write("\n")
    return target_path


def _filter_kwargs(data: dict[str, Any], model: type) -> dict[str, Any]:
    allowed = set(model.__dataclass_fields__)
    return {key: value for key, value in data.items() if key in allowed}


def _require_config_format(data: dict[str, Any]) -> None:
    required_config_fields = set(DatasetConfig.__dataclass_fields__) - {
        "config_version",
        "val_count",
        "subset_totals",
        "save_masks",
    }
    missing_config_fields = sorted(required_config_fields - set(data))
    if missing_config_fields:
        missing = ", ".join(missing_config_fields)
        raise ValueError(f"Config file format is invalid. Missing required field(s): {missing}.")

    render_data = data.get("render")
    if not isinstance(render_data, dict):
        raise ValueError("Config file format is invalid. Field 'render' must be an object.")

    required_render_fields = set(ShapeRenderConfig.__dataclass_fields__)
    missing_render_fields = sorted(required_render_fields - set(render_data))
    if missing_render_fields:
        missing = ", ".join(f"render.{field}" for field in missing_render_fields)
        raise ValueError(f"Config file format is invalid. Missing required field(s): {missing}.")
