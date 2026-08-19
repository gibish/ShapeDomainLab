"""Summarise class-level error patterns from saved prediction files."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse errors in robustness-matrix predictions.")
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    counter: Counter[tuple[str, str, str, str, str]] = Counter()
    for path in args.runs.glob("*/*/seed-*/predictions-*.csv"):
        model, training_domain, _ = path.parent.relative_to(args.runs).parts
        evaluation_domain = path.stem.removeprefix("predictions-")
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row["target"] != row["prediction"]:
                    counter[(model, training_domain, evaluation_domain, row["target"], row["prediction"])] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("model", "training_domain", "evaluation_domain", "true_class", "predicted_class", "error_count")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for key, count in counter.most_common():
            writer.writerow(dict(zip(fields, (*key, count))))
    lines = ["# Аналіз помилок класифікації", "", "Наведено найчастіші неправильні переходи між класами, агреговані за п'ятьма зернами (максимум 2 500 прогнозів для кожної пари тренувального і тестового доменів).", "", "| Модель | Навчання | Тест | Істинний клас → прогноз | Кількість помилок |", "|---|---|---|---|---:|"]
    for (model, train, test, true, predicted), count in counter.most_common(20):
        lines.append(f"| {model} | {train} | {test} | {true} → {predicted} | {count} |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysed {sum(counter.values())} errors.")


if __name__ == "__main__":
    main()
