from pathlib import Path

from src.core.config_io import load_config_from_json


CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "configs"


def test_controlled_benchmark_configs_are_valid_and_balanced() -> None:
    config_paths = sorted(CONFIG_DIR.glob("*.json"))

    assert [path.stem for path in config_paths] == [
        "clean",
        "clipping",
        "composite",
        "geometry",
        "noise",
        "rotation",
    ]

    configs = {path.stem: load_config_from_json(path) for path in config_paths}
    clean = configs["clean"]
    for config in configs.values():
        assert config.classes == clean.classes
        assert config.image_size == clean.image_size == 64
        assert (config.train_count, config.val_count, config.test_count) == (1000, 200, 500)
        assert config.file_format == "png"
        assert config.save_masks
        assert config.render.color_mode == "grayscale"
        assert config.render.fill_mode == "outline_fill"
        assert config.render.background_range == (160, 201)
        assert config.render.foreground_range == (0, 51)
        assert config.render.line_width_range == (1, 4)


def test_controlled_benchmark_configs_vary_their_named_factors() -> None:
    configs = {path.stem: load_config_from_json(path) for path in CONFIG_DIR.glob("*.json")}
    clean = configs["clean"].render

    assert configs["rotation"].render.rotation_range == (-45.0, 45.0)
    assert configs["geometry"].render.scale_range == (0.7, 1.3)
    assert configs["geometry"].render.center_offset_range == (0.0, 12.0)
    assert configs["noise"].render.noise_level == 20
    assert configs["clipping"].render.edge_clipping
    assert configs["composite"].render.rotation_range == (-45.0, 45.0)
    assert configs["composite"].render.scale_range == (0.7, 1.3)
    assert configs["composite"].render.noise_level == 20
    assert clean.rotation_range == (0.0, 0.0)
    assert not clean.edge_clipping
