"""Produce research-ready summaries for a completed robustness matrix."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_results(runs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(runs_root.glob("*/*/seed-*/metrics-*.json")):
        model, training_domain, seed = path.parent.relative_to(runs_root).parts
        evaluation_domain = path.stem.removeprefix("metrics-")
        if evaluation_domain == "validation":
            continue
        values = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": model, "training_domain": training_domain,
            "evaluation_domain": evaluation_domain, "seed": int(seed.removeprefix("seed-")),
            **{metric: float(values[metric]) for metric in METRICS},
        })
    return rows


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["training_domain"]), str(row["evaluation_domain"]))].append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "training_domain", "evaluation_domain", "n"]
    fields += [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            item: dict[str, object] = dict(zip(("model", "training_domain", "evaluation_domain"), key))
            item["n"] = len(group)
            for metric in METRICS:
                item[f"{metric}_mean"], item[f"{metric}_std"] = _mean_std([float(r[metric]) for r in group])
            writer.writerow(item)


def write_report(rows: list[dict[str, object]], output: Path) -> None:
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)

    lines = ["# Статистичне узагальнення матриці робастності", "", "Усі показники є середнім ± стандартним відхиленням за п'ятьма незалежними зернами. Метрики тестової вибірки не використовувалися для вибору архітектури чи кількості епох.", ""]
    for model, model_rows in sorted(by_model.items()):
        in_domain = [float(r["macro_f1"]) for r in model_rows if r["training_domain"] == r["evaluation_domain"]]
        cross_domain = [float(r["macro_f1"]) for r in model_rows if r["training_domain"] != r["evaluation_domain"]]
        a_mean, a_std = _mean_std(in_domain)
        c_mean, c_std = _mean_std(cross_domain)
        lines += [f"## {model}", "", f"- Внутрішньодоменний macro-F1: {a_mean:.4f} ± {a_std:.4f}.", f"- Міждоменний macro-F1: {c_mean:.4f} ± {c_std:.4f}.", f"- Абсолютне зниження при доменному зсуві: {a_mean - c_mean:.4f}.", ""]
    lines += ["## Повна таблиця macro-F1", ""]
    for model, model_rows in sorted(by_model.items()):
        domains = sorted({str(r["training_domain"]) for r in model_rows})
        lines += [f"### {model}", "", "| Навчання \\ Тест | " + " | ".join(domains) + " |", "|---|" + "---|" * len(domains)]
        for train in domains:
            values = []
            for test in domains:
                group = [float(r["macro_f1"]) for r in model_rows if r["training_domain"] == train and r["evaluation_domain"] == test]
                mean, std = _mean_std(group)
                values.append(f"{mean:.3f} ± {std:.3f}")
            lines.append("| " + train + " | " + " | ".join(values) + " |")
        lines.append("")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmaps(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        domains = sorted({str(row["training_domain"]) for row in model_rows})
        matrix = np.array([[statistics.mean(float(r["macro_f1"]) for r in model_rows if r["training_domain"] == train and r["evaluation_domain"] == test) for test in domains] for train in domains])
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(domains)), domains, rotation=35, ha="right")
        axis.set_yticks(range(len(domains)), domains)
        axis.set_xlabel("Тестовий домен")
        axis.set_ylabel("Тренувальний домен")
        axis.set_title(f"{model}: середній macro-F1 (n=5)")
        for i in range(len(domains)):
            for j in range(len(domains)):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.55 else "black")
        figure.colorbar(image, ax=axis, label="macro-F1")
        figure.tight_layout()
        figure.savefig(output.parent / f"{output.stem}-{model}-macro-f1.png", dpi=300)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a completed ShapeDomainLab robustness matrix.")
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = load_results(args.runs)
    if len(rows) != 360:
        raise ValueError(f"Expected 360 test metric files, found {len(rows)}.")
    write_summary(rows, args.output)
    write_report(rows, args.output)
    write_heatmaps(rows, args.output)
    print(f"Analysed {len(rows)} test metric files.")


if __name__ == "__main__":
    main()
