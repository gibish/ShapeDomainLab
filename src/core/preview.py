from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import DatasetConfig
from .shapes import render_shape
from .validation import require_valid_config


@dataclass(frozen=True)
class PreviewImage:
    class_name: str
    image: Image.Image


def generate_preview_images(config: DatasetConfig, limit: int = 8) -> list[PreviewImage]:
    if limit < 1:
        return []
    if not config.classes:
        raise ValueError("Select at least one shape class.")

    require_valid_config(config)
    rng = np.random.default_rng(config.seed)
    images: list[PreviewImage] = []

    for index in range(limit):
        class_name = config.classes[index % len(config.classes)]
        image = render_shape(class_name, rng, config.image_size, config.render)
        images.append(PreviewImage(class_name=class_name, image=image))

    return images
