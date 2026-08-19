import argparse
import csv
import platform
import sys
from pathlib import Path

from experiments.common import build_model, configure_seeds, dataset_channels, evaluate_model, image_transform, require_torch, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reproducible ShapeDomainLab classifier.")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root containing train/ and val/.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for the run artifacts.")
    parser.add_argument("--model", choices=("tiny_cnn", "resnet18"), default="tiny_cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a Torch device such as cuda.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run from its epoch checkpoint.")
    return parser.parse_args()


def _save_torch(path: Path, payload, torch) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    torch, torchvision = require_torch()
    configure_seeds(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    transform = image_transform(args.dataset, torchvision)
    train_set = torchvision.datasets.ImageFolder(args.dataset / "train", transform=transform)
    val_set = torchvision.datasets.ImageFolder(args.dataset / "val", transform=transform)
    if train_set.classes != val_set.classes:
        raise ValueError("Train and validation class mappings differ.")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    model = build_model(args.model, dataset_channels(args.dataset), len(train_set.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "training_checkpoint.pt"
    history_path = args.output / "history.json"
    status_path = args.output / "training_status.json"
    best_score = -1.0
    history: list[dict] = []
    start_epoch = 1
    if args.resume:
        if not state_path.is_file():
            raise ValueError(f"Cannot resume because the epoch checkpoint does not exist: {state_path}")
        # The data-loader generator state must remain a CPU ByteTensor even
        # when model parameters are restored for CUDA execution.
        state = torch.load(state_path, map_location="cpu")
        if state["model"] != args.model or state["seed"] != args.seed:
            raise ValueError("The resume checkpoint does not match the requested model or seed.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        generator.set_state(state["data_loader_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        best_score = float(state["best_score"])
        history = state["history"]
        start_epoch = int(state["completed_epoch"]) + 1

    run_config = {
        **vars(args),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "device": str(device),
        "class_names": train_set.classes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }
    write_json(args.output / "run_config.json", run_config)
    write_json(status_path, {"state": "in_progress", "completed_epochs": start_epoch - 1, "target_epochs": args.epochs})

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics, _ = evaluate_model(model, val_loader, device, train_set.classes)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), **metrics})
        if metrics["macro_f1"] >= best_score:
            best_score = metrics["macro_f1"]
            _save_torch(
                args.output / "model.pt",
                {"model": args.model, "channels": dataset_channels(args.dataset), "class_names": train_set.classes, "state_dict": model.state_dict()},
                torch,
            )
        write_json(history_path, history)
        _save_torch(
            state_path,
            {
                "model": args.model,
                "seed": args.seed,
                "completed_epoch": epoch,
                "best_score": best_score,
                "history": history,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "data_loader_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            torch,
        )
        write_json(status_path, {"state": "in_progress", "completed_epochs": epoch, "target_epochs": args.epochs})
    checkpoint = torch.load(args.output / "model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, predictions = evaluate_model(model, val_loader, device, train_set.classes)
    write_json(args.output / "metrics-validation.json", metrics)
    write_json(history_path, history)
    with (args.output / "predictions-validation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=predictions[0].keys())
        writer.writeheader(); writer.writerows(predictions)
    write_json(status_path, {"state": "completed", "completed_epochs": args.epochs, "target_epochs": args.epochs})
    print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
