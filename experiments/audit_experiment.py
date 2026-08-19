"""Audit completeness and provenance of a completed robustness matrix."""

import argparse
import json
import platform
from pathlib import Path


DOMAINS = ("clean", "rotation", "geometry", "noise", "clipping", "composite")
MODELS = ("tiny_cnn", "resnet18")
SEEDS = (1, 2, 3, 4, 5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed robustness experiment.")
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    dataset_summary: dict[str, object] = {}
    for domain in DOMAINS:
        report_path = args.datasets / domain / "quality_report.json"
        if not report_path.is_file():
            errors.append(f"Відсутній quality_report.json: {domain}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stats = report.get("statistics", {})
        exact = stats.get("exact_duplicate_groups", [])
        cross = stats.get("cross_subset_exact_duplicate_groups", [])
        if not report.get("passed") or exact or cross:
            errors.append(f"Порушено контракт якості набору: {domain}")
        dataset_summary[domain] = {"passed": report.get("passed"), "images": stats.get("total_images"), "exact_duplicate_groups": len(exact), "cross_subset_exact_duplicate_groups": len(cross), "manifest_algorithm": stats.get("manifest", {}).get("algorithm")}
    completed = 0
    evaluations = 0
    for model in MODELS:
        for train in DOMAINS:
            for seed in SEEDS:
                run = args.runs / model / train / f"seed-{seed}"
                status = run / "training_status.json"
                if not status.is_file() or json.loads(status.read_text(encoding="utf-8")).get("state") != "completed" or not (run / "model.pt").is_file():
                    errors.append(f"Неповне навчання: {run}")
                    continue
                completed += 1
                for test in DOMAINS:
                    if not (run / f"metrics-{test}.json").is_file() or not (run / f"predictions-{test}.csv").is_file():
                        errors.append(f"Неповне оцінювання: {run}, {test}")
                    else:
                        evaluations += 1
    progress = json.loads((args.runs / "experiment_progress.json").read_text(encoding="utf-8"))
    audit = {"passed": not errors, "matrix": {"completed_trainings": completed, "expected_trainings": 60, "completed_evaluations": evaluations, "expected_evaluations": 360, "progress_state": progress.get("state")}, "datasets": dataset_summary, "environment": {"python": platform.python_version(), "platform": platform.platform()}, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Аудит цілісності експерименту", "", f"**Статус:** {'пройдено' if audit['passed'] else 'не пройдено'}.", "", f"- Навчання: {completed}/60.", f"- Оцінювання: {evaluations}/360.", f"- Стан диспетчера: `{progress.get('state')}`.", "", "| Домен | Зображень | Точні дублікати | Міжпідмножинні точні дублікати |", "|---|---:|---:|---:|"]
    lines += [f"| {domain} | {item['images']} | {item['exact_duplicate_groups']} | {item['cross_subset_exact_duplicate_groups']} |" for domain, item in dataset_summary.items()]
    if errors:
        lines += ["", "## Помилки", ""] + [f"- {error}" for error in errors]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit {'passed' if audit['passed'] else 'failed'}.")


if __name__ == "__main__":
    main()
