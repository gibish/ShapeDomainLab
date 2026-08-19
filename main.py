import argparse
from pathlib import Path

from src.core.config import SUPPORTED_COLOR_MODES, SUPPORTED_FILL_MODES, DatasetConfig, ShapeRenderConfig
from src.core.config_io import load_config_from_json, save_config_template
from src.core.analysis import analyze_dataset
from src.core.generator import generate_dataset
from src.core.quality import validate_dataset
from src.core.summary import dataset_summary, estimate_dataset_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a ShapeDomainLab dataset.")
    parser.add_argument("--config", help="Load generation settings from a generation_config.json file.")
    parser.add_argument("--save-config", help="Save the selected generation settings to a JSON file and exit.")
    parser.add_argument("--summary", help="Print a summary for an existing generated dataset and exit.")
    parser.add_argument("--analyze", help="Write an analysis report and figures for an existing generated dataset.")
    parser.add_argument("--estimate", action="store_true", help="Print an estimated output size before generation.")
    parser.add_argument("--output", default="datasets/shapes", help="Output dataset directory.")
    parser.add_argument("--image-size", type=int, default=64, help="Generated image width and height.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--dataset-count", type=int, help="Total images across train, val, and test.")
    split_group.add_argument("--train-count", type=int, help="Train images per class in manual split mode.")
    parser.add_argument("--val-count", type=int, help="Validation images per class in manual split mode.")
    parser.add_argument("--test-count", type=int, help="Test images per class in manual split mode.")
    parser.add_argument(
        "--split-ratio",
        choices=("70/15/15", "80/10/10"),
        default="70/15/15",
        help="Automatic train/val/test ratio used with --dataset-count.",
    )
    parser.add_argument("--format", default="png", help="Image file format.")
    parser.add_argument("--save-masks", action="store_true", help="Save binary PNG segmentation masks in masks/.")
    parser.add_argument("--color-mode", choices=SUPPORTED_COLOR_MODES, default="grayscale", help="Image color mode.")
    parser.add_argument(
        "--background-color",
        nargs=3,
        type=int,
        default=(180, 180, 180),
        metavar=("R", "G", "B"),
        help="RGB background color used when --color-mode rgb.",
    )
    parser.add_argument(
        "--foreground-color",
        nargs=3,
        type=int,
        default=(0, 0, 0),
        metavar=("R", "G", "B"),
        help="RGB shape color used when --color-mode rgb.",
    )
    parser.add_argument("--fill-mode", choices=SUPPORTED_FILL_MODES, default="outline", help="Shape fill mode.")
    parser.add_argument("--rotation-min", type=float, default=0.0, help="Minimum rotation angle in degrees.")
    parser.add_argument("--rotation-max", type=float, default=0.0, help="Maximum rotation angle in degrees.")
    parser.add_argument("--scale-min", type=float, default=1.0, help="Minimum shape scale factor.")
    parser.add_argument("--scale-max", type=float, default=1.0, help="Maximum shape scale factor.")
    parser.add_argument("--center-offset-min", type=float, default=0.0, help="Minimum center offset in pixels.")
    parser.add_argument("--center-offset-max", type=float, default=0.0, help="Maximum center offset in pixels.")
    parser.add_argument(
        "--edge-clipping",
        action="store_true",
        help="Allow shapes to be partially clipped by the image edge.",
    )
    parser.add_argument(
        "--min-visible-ratio",
        type=float,
        default=0.6,
        help="Minimum visible part of a clipped shape, from 0.6 to 1.0.",
    )
    parser.add_argument("--noise-level", type=int, default=0, help="Uniform background noise level.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["circle", "octagon", "rectangle", "square", "triangle"],
        help="Shape classes to generate.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    if args.config:
        return load_config_from_json(args.config)

    if args.dataset_count is not None:
        if args.val_count is not None or args.test_count is not None:
            raise ValueError("--dataset-count cannot be combined with --val-count or --test-count.")
        train_ratio, val_ratio, test_ratio = (int(value) for value in args.split_ratio.split("/"))
        val_total = args.dataset_count * val_ratio // 100
        test_total = val_total
        train_total = args.dataset_count - val_total - test_total
        subset_totals = {"train": train_total, "val": val_total, "test": test_total}
        train_count = max(1, train_total // max(1, len(args.classes)))
        val_count = max(1, val_total // max(1, len(args.classes)))
        test_count = max(1, test_total // max(1, len(args.classes)))
    else:
        train_count = args.train_count if args.train_count is not None else 10
        val_count = args.val_count if args.val_count is not None else 10
        test_count = args.test_count if args.test_count is not None else 10
        subset_totals = None

    return DatasetConfig(
        output_dir=Path(args.output),
        image_size=args.image_size,
        classes=tuple(args.classes),
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        subset_totals=subset_totals,
        file_format=args.format,
        save_masks=args.save_masks,
        seed=args.seed,
        render=ShapeRenderConfig(
            color_mode=args.color_mode,
            background_color=tuple(args.background_color),
            foreground_color=tuple(args.foreground_color),
            fill_mode=args.fill_mode,
            rotation_range=(args.rotation_min, args.rotation_max),
            scale_range=(args.scale_min, args.scale_max),
            center_offset_range=(args.center_offset_min, args.center_offset_max),
            edge_clipping=args.edge_clipping,
            min_visible_ratio=args.min_visible_ratio,
            noise_level=args.noise_level,
        ),
    )


def main() -> None:
    args = parse_args()
    if args.summary:
        summary = dataset_summary(args.summary)
        print(f"Dataset: {summary.output_dir}")
        print(f"Exists: {summary.exists}")
        print(f"Total images: {summary.total_images}")
        if summary.subsets:
            for subset, class_counts in summary.subsets.items():
                counts = ", ".join(f"{name}={count}" for name, count in class_counts.items())
                print(f"{subset}: {counts}")
        if summary.image_formats:
            formats = ", ".join(f"{name}={count}" for name, count in summary.image_formats.items())
            print(f"Formats: {formats}")
        missing = [name for name, present in summary.service_files.items() if not present]
        if missing:
            print(f"Missing service files: {', '.join(missing)}")
        return

    if args.analyze:
        analysis_dir = analyze_dataset(args.analyze)
        print(f"Dataset analysis: {analysis_dir}")
        return

    config = config_from_args(args)
    if args.save_config:
        config_path = save_config_template(args.save_config, config)
        print(f"Config saved: {config_path}")
        return

    if args.estimate:
        estimate = estimate_dataset_size(config)
        print(f"Estimated images: {estimate.total_images}")
        print(f"Estimated size: {estimate.estimated_total_bytes} bytes")

    def confirm_overwrite(output_dir: Path) -> bool:
        answer = input(
            f"Output directory already contains a generated dataset:\n{output_dir}\n"
            "Clear it and generate a new dataset? [y/N]: "
        )
        return answer.strip().lower() in {"y", "yes"}

    output_dir = generate_dataset(config, confirm_overwrite=confirm_overwrite)
    print(f"Dataset generated: {output_dir}")
    report = validate_dataset(output_dir, config)
    if report.passed:
        print("Quality check: passed")
    else:
        print("Quality check: failed")
        for error in report.errors:
            print(f"- {error}")
    for warning in report.warnings:
        print(f"Quality warning: {warning}")
    print(f"Quality reports: {output_dir / 'quality_report.json'}, {output_dir / 'quality_report.md'}")


if __name__ == "__main__":
    main()
