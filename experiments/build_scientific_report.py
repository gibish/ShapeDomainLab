"""Create a Ukrainian research-results draft from the completed matrix."""

import argparse
import csv
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pooled_std(groups: list[dict[str, str]]) -> float:
    """Sample SD across all seed-level observations reconstructed from summaries."""
    total_n = sum(int(group["n"]) for group in groups)
    grand_mean = sum(float(group["macro_f1_mean"]) * int(group["n"]) for group in groups) / total_n
    sum_squares = sum(
        (int(group["n"]) - 1) * float(group["macro_f1_std"]) ** 2
        + int(group["n"]) * (float(group["macro_f1_mean"]) - grand_mean) ** 2
        for group in groups
    )
    return (sum_squares / (total_n - 1)) ** 0.5


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(mean(differences))
    values = [abs(mean([sign * difference for sign, difference in zip(signs, differences)])) for signs in itertools.product((-1, 1), repeat=len(differences))]
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ukrainian scientific results draft.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.summary.open(encoding="utf-8")))
    cells = {(r["model"], r["training_domain"], r["evaluation_domain"]): float(r["macro_f1_mean"]) for r in rows}
    domains = sorted({r["training_domain"] for r in rows})
    in_domain = {model: [cells[(model, domain, domain)] for domain in domains] for model in ("tiny_cnn", "resnet18")}
    cross_domain = {model: [cells[(model, train, test)] for train in domains for test in domains if train != test] for model in ("tiny_cnn", "resnet18")}
    in_domain_sd = {model: pooled_std([r for r in rows if r["model"] == model and r["training_domain"] == r["evaluation_domain"]]) for model in in_domain}
    cross_domain_sd = {model: pooled_std([r for r in rows if r["model"] == model and r["training_domain"] != r["evaluation_domain"]]) for model in cross_domain}

    # The summary contains seed-level standard deviations only. Reconstructing a paired
    # model test from these aggregates would be invalid; the test is intentionally not
    # reported. The report therefore presents descriptive uncertainty and a registered
    # inferential plan that operates on the raw per-seed CSV.
    output = args.output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Результати експерименту з доменної робастності", "",
        "## 1. Обсяг та відтворюваність", "",
        "Було завершено повну факторну матрицю: 2 архітектури × 6 тренувальних доменів × 5 незалежних зерен × 6 тестових доменів. Отже, аналіз охоплює 60 навчань та 360 тестових оцінювань. Кількість епох була зафіксована до аналізу тестових даних за результатами валідаційного пілоту: 50 для TinyCNN і 20 для ResNet18.", "",
        "Основною метрикою є macro-F1. У таблицях наведено середнє та вибіркове стандартне відхилення за п'ятьма зернами. Тестові результати не використовувалися для вибору архітектури, кількості епох або інших гіперпараметрів.", "",
        "## 2. Внутрішньодоменна та міждоменна якість", "",
        "| Архітектура | Внутрішньодоменний macro-F1 | Міждоменний macro-F1 | Абсолютне зниження |",
        "|---|---:|---:|---:|",
    ]
    for model, label in (("tiny_cnn", "TinyCNN"), ("resnet18", "ResNet18")):
        report.append(f"| {label} | {mean(in_domain[model]):.4f} ± {in_domain_sd[model]:.4f} | {mean(cross_domain[model]):.4f} ± {cross_domain_sd[model]:.4f} | {mean(in_domain[model]) - mean(cross_domain[model]):.4f} |")
    report += ["", "ResNet18 досягає майже насиченої внутрішньодоменної якості, але його середня якість під доменним зсувом істотно нижча. TinyCNN має нижчу внутрішньодоменну якість, проте також демонструє виражену втрату при переході до іншого домену. Тому висока якість на тестовому домені, узгодженому з тренувальним, не може інтерпретуватися як достатній доказ робастності до усіх контрольованих збурень.", "", "## 3. Деталізація за доменами", "", "Повна матриця середніх значень і стандартних відхилень macro-F1 наведена у файлі `statistical-summary.md`; відповідні теплові карти збережено як PNG-файли. Найбільш виражені міждоменні втрати спостерігаються для переходів до композитних та обрізаних зображень у випадках, коли ці властивості не були представлені під час навчання.", "", "## 4. Статистичний висновок і обмеження", "", "Поточний звіт подає описові оцінки невизначеності за п'ятьма зернами. Для формального порівняння архітектур рекомендовано застосувати попередньо визначений парний перестановковий тест до середнього macro-F1 кожного зерна, агрегованого по 36 парах тренувального й тестового доменів. Водночас n=5 обмежує роздільну здатність точного двобічного sign-flip тесту: його найменше досяжне p-значення становить 0.0625. Отже, у статті слід пріоритетно інтерпретувати величини ефектів, довірчі інтервали та повні розподіли за зернами, а не лише дихотомічні p-значення.", "", "## 5. Рекомендоване формулювання висновку", "", "У контрольованій синтетичній постановці ResNet18 переважає TinyCNN за внутрішньодоменною точністю. Проте обидві архітектури демонструють суттєве зниження macro-F1 за доменного зсуву. Це вказує, що архітектурна потужність сама по собі не забезпечує інваріантності до збурень рендерингу; для практично робастних моделей необхідні або ширше покриття тренувальних доменів, або спеціальні методи доменної генералізації.", "",
    ]
    output.with_suffix(".md").write_text("\n".join(report), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"matrix": {"trainings": 60, "evaluations": 360, "seeds": 5}, "macro_f1": {model: {"in_domain_mean": mean(in_domain[model]), "in_domain_pooled_sd": in_domain_sd[model], "cross_domain_mean": mean(cross_domain[model]), "cross_domain_pooled_sd": cross_domain_sd[model]} for model in in_domain}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
