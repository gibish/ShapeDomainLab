import argparse
import csv
import json
from pathlib import Path


def collect_rows(runs_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(runs_root.rglob("metrics-*.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"run": str(metrics_path.parent.relative_to(runs_root)), "evaluation": metrics_path.stem.removeprefix("metrics-"), "accuracy": str(data["accuracy"]), "balanced_accuracy": str(data["balanced_accuracy"]), "macro_f1": str(data["macro_f1"]), "sample_count": str(data["sample_count"])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ShapeDomainLab experiment metrics.")
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = collect_rows(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("run", "evaluation", "accuracy", "balanced_accuracy", "macro_f1", "sample_count"))
        writer.writeheader(); writer.writerows(rows)
    markdown = ["# Experiment Results", "", "| Run | Evaluation | Accuracy | Balanced accuracy | Macro-F1 | Samples |", "|---|---:|---:|---:|---:|---:|"]
    markdown.extend(f"| {row['run']} | {row['evaluation']} | {row['accuracy']} | {row['balanced_accuracy']} | {row['macro_f1']} | {row['sample_count']} |" for row in rows)
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Aggregated {len(rows)} metric files.")


if __name__ == "__main__":
    main()
