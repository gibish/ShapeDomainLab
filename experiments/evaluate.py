import argparse
import csv
from pathlib import Path

from experiments.common import build_model, dataset_channels, evaluate_model, image_transform, require_torch, write_json


def _write_predictions(path: Path, predictions: list[dict]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=predictions[0].keys())
        writer.writeheader(); writer.writerows(predictions)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ShapeDomainLab classifier.")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root containing test/.")
    parser.add_argument("--run", required=True, type=Path, help="Training run directory containing model.pt.")
    parser.add_argument("--name", required=True, help="Evaluation domain name, for example rotation.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch, torchvision = require_torch()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.run / "model.pt", map_location=device)
    test_set = torchvision.datasets.ImageFolder(args.dataset / "test", transform=image_transform(args.dataset, torchvision))
    if test_set.classes != checkpoint["class_names"]:
        raise ValueError("Test dataset class mapping differs from the training checkpoint.")
    model = build_model(checkpoint["model"], dataset_channels(args.dataset), len(test_set.classes)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, predictions = evaluate_model(model, torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False), device, test_set.classes)
    _write_predictions(args.run / f"predictions-{args.name}.csv", predictions)
    write_json(args.run / f"metrics-{args.name}.json", {**metrics, "dataset": str(args.dataset), "evaluation_name": args.name})
    print(f"{args.name} macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
