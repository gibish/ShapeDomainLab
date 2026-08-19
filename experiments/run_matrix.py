"""Run or resume the controlled cross-domain robustness matrix."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.common import write_json


DOMAINS = ("clean", "rotation", "geometry", "noise", "clipping", "composite")
EPOCHS = {"tiny_cnn": 50, "resnet18": 20}


def _valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def training_complete(run_dir: Path) -> bool:
    status_path = run_dir / "training_status.json"
    if not status_path.is_file() or not (run_dir / "model.pt").is_file() or not _valid_json(status_path):
        return False
    return json.loads(status_path.read_text(encoding="utf-8")).get("state") == "completed"


def evaluation_complete(run_dir: Path, domain: str) -> bool:
    return (
        (run_dir / f"predictions-{domain}.csv").is_file()
        and (run_dir / f"metrics-{domain}.json").is_file()
        and _valid_json(run_dir / f"metrics-{domain}.json")
    )


def _require_valid_datasets(datasets_root: Path) -> None:
    for domain in DOMAINS:
        report_path = datasets_root / domain / "quality_report.json"
        if not _valid_json(report_path) or not json.loads(report_path.read_text(encoding="utf-8")).get("passed"):
            raise ValueError(f"Dataset {domain!r} is missing a passing quality_report.json: {report_path}")


def _run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume the ShapeDomainLab robustness matrix.")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets/benchmarks"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/robustness-matrix"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    _require_valid_datasets(args.datasets_root)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    total_trainings = len(EPOCHS) * len(DOMAINS) * len(args.seeds)
    total_evaluations = total_trainings * len(DOMAINS)
    completed_trainings = 0
    completed_evaluations = 0
    progress_path = args.runs_root / "experiment_progress.json"

    for model, epochs in EPOCHS.items():
        for training_domain in DOMAINS:
            for seed in args.seeds:
                run_dir = args.runs_root / model / training_domain / f"seed-{seed}"
                if not training_complete(run_dir):
                    command = [
                        sys.executable, "-m", "experiments.train", "--dataset", str(args.datasets_root / training_domain),
                        "--output", str(run_dir), "--model", model, "--epochs", str(epochs), "--seed", str(seed), "--device", args.device,
                    ]
                    if (run_dir / "training_checkpoint.pt").is_file():
                        command.append("--resume")
                    _run(command)
                completed_trainings += 1
                write_json(progress_path, {"state": "in_progress", "completed_trainings": completed_trainings, "total_trainings": total_trainings, "completed_evaluations": completed_evaluations, "total_evaluations": total_evaluations})

                for evaluation_domain in DOMAINS:
                    if not evaluation_complete(run_dir, evaluation_domain):
                        _run([
                            sys.executable, "-m", "experiments.evaluate", "--dataset", str(args.datasets_root / evaluation_domain),
                            "--run", str(run_dir), "--name", evaluation_domain, "--device", args.device,
                        ])
                    completed_evaluations += 1
                    write_json(progress_path, {"state": "in_progress", "completed_trainings": completed_trainings, "total_trainings": total_trainings, "completed_evaluations": completed_evaluations, "total_evaluations": total_evaluations})

    write_json(progress_path, {"state": "completed", "completed_trainings": total_trainings, "total_trainings": total_trainings, "completed_evaluations": total_evaluations, "total_evaluations": total_evaluations})
    print("Robustness matrix completed.")


if __name__ == "__main__":
    main()
