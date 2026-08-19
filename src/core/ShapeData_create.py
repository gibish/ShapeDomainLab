from .config import DatasetConfig
from .generator import generate_dataset


def create_all_imgs() -> None:
    generate_dataset(DatasetConfig(train_count=15000, test_count=2500))
