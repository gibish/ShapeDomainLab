import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def require_torch():
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "Experiment dependencies are missing. Install them with "
            "`pip install -r requirements-experiments.txt`."
        ) from exc
    return torch, torchvision


def configure_seeds(seed: int) -> None:
    torch, _ = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_channels(dataset_root: Path) -> int:
    config = json.loads((dataset_root / "generation_config.json").read_text(encoding="utf-8"))
    return 3 if config["render"]["color_mode"] == "rgb" else 1


def image_transform(dataset_root: Path, torchvision):
    """Return a tensor transform with the channel count declared by the dataset."""
    channels = dataset_channels(dataset_root)
    transforms = []
    if channels == 1:
        # ImageFolder's default loader converts source files to RGB. Convert it
        # back explicitly so grayscale datasets match one-channel model inputs.
        transforms.append(torchvision.transforms.Grayscale(num_output_channels=1))
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)


def build_model(name: str, channels: int, class_count: int):
    torch, torchvision = require_torch()
    if name == "tiny_cnn":
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, class_count),
        )
    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None, num_classes=class_count)
        model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    raise ValueError(f"Unsupported model '{name}'. Choose tiny_cnn or resnet18.")


def metrics_from_predictions(targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have equal length.")
    if not targets:
        return {"sample_count": 0, "accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_id, class_name in enumerate(class_names):
        tp = sum(target == class_id and prediction == class_id for target, prediction in zip(targets, predictions))
        fp = sum(target != class_id and prediction == class_id for target, prediction in zip(targets, predictions))
        fn = sum(target == class_id and prediction != class_id for target, prediction in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {"support": sum(target == class_id for target in targets), "precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "sample_count": len(targets),
        "accuracy": sum(target == prediction for target, prediction in zip(targets, predictions)) / len(targets),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
    }


def evaluate_model(model, loader, device, class_names: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch, _ = require_torch()
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            batch_targets = labels.tolist()
            for index, (target, prediction) in enumerate(zip(batch_targets, batch_predictions)):
                rows.append({"sample_index": offset + index, "target_id": target, "target": class_names[target], "prediction_id": prediction, "prediction": class_names[prediction]})
            offset += len(batch_targets)
            targets.extend(batch_targets)
            predictions.extend(batch_predictions)
    return metrics_from_predictions(targets, predictions, class_names), rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
