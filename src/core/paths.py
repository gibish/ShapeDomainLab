from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_output_dir(output_dir: Path | str) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return project_root() / path


def subset_dir(output_dir: Path, subset: str) -> Path:
    return output_dir / subset


def class_dir(output_dir: Path, subset: str, class_name: str) -> Path:
    return subset_dir(output_dir, subset) / class_name


def mask_dir(output_dir: Path, subset: str, class_name: str) -> Path:
    return output_dir / "masks" / subset / class_name
