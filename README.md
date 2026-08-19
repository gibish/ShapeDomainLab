# ShapeDomainLab

ShapeDomainLab is a Python desktop and command-line application for reproducibly generating, documenting, auditing, and experimentally using controlled synthetic domains of simple geometric shapes.

Україномовний посібник користувача та методологічна документація для наукових експериментів доступні у [DOCUMENTATION_UA.md](DOCUMENTATION_UA.md).

The current version generates grayscale or RGB images for these classes:

- `circle`
- `octagon`
- `rectangle`
- `square`
- `triangle`

Generated datasets are compatible with `torchvision.datasets.ImageFolder`-style folder layouts.

## Current Version

Release version: `1.0`

## Dataset Structure

ShapeDomainLab creates `train`, `val`, and `test` subsets:

```text
dataset/
  train/
    circle/
    octagon/
    rectangle/
    square/
    triangle/
  val/
    circle/
    octagon/
    rectangle/
    square/
    triangle/
  test/
    circle/
    octagon/
    rectangle/
    square/
    triangle/
  masks/  (optional; binary PNG masks matching the image subsets)
    train/<class>/
    val/<class>/
    test/<class>/
  metadata.csv
  class_mapping.json
  generation_config.json
  log.txt
  README.md
```

The `train`, `val`, and `test` subsets are generated independently. Use `train` for model training, `val` for hyperparameter tuning and model selection, and `test` only for final evaluation.

When a requested subset total cannot be divided evenly among the selected classes, the remainder is distributed one image at a time across the classes. Therefore individual class folders may contain different numbers of images, while the total for each subset remains exact.

## Service Files

Each generated dataset includes:

- `metadata.csv`: per-image labels, independent sample seed, sampled rendering parameters, geometric dimensions and image-coordinate bounds, clipping information, and the application, Python, NumPy, and Pillow versions used for generation. The metadata schema version is recorded in every row.
- `class_mapping.json`: mapping between class names and numeric IDs.
- `generation_config.json`: configuration used for generation.
- `dataset_manifest.json`: SHA-256 integrity manifest for generated artifacts.
- `log.txt`: short generation summary.
- `quality_report.json` and `quality_report.md`: results of the post-generation quality validation.
- `README.md`: dataset-level description.
- `masks/` (optional): binary PNG segmentation masks; the main `train/`, `val/`, and `test/` directories remain ImageFolder-compatible.

### Metadata schema

`metadata.csv` uses schema version `2.1`. Each row records one generated image.

| Field group | Fields | Meaning |
|---|---|---|
| Provenance | `metadata_schema_version`, `application_version`, `python_version`, `numpy_version`, `pillow_version` | Versions needed to interpret and reproduce the generated artifact. |
| Identity and label | `filename`, `mask_filename`, `subset`, `class`, `class_id`, `image_size`, `shape_type`, `seed`, `status` | Image location, optional binary-mask location, dataset partition, class label, independent sample seed, and generation status. |
| Rendering | `background`, `foreground`, `line_width`, `scale`, `rotation_degrees`, `color_mode`, `fill_mode`, `noise_level`, `margin` | Sampled or configured rendering parameters. |
| Geometry | `radius`, `shape_width`, `shape_height`, `shape_area` | Nominal geometric parameters before rasterization. The radius is recorded for circles and octagons where applicable. |
| Bounds and clipping | `center_x`, `center_y`, `bbox_left`, `bbox_top`, `bbox_right`, `bbox_bottom`, `edge_clipping`, `min_visible_ratio`, `clipping_side` | Shape position and its rendered geometric bounds in image coordinates. Bounds may extend beyond the image when edge clipping is enabled. |

`shape_width`, `shape_height`, and `shape_area` describe the generated geometric primitive; they are not measurements of rasterized foreground pixels. `bbox_*` values are the bounds after rotation and placement, before the image frame clips any part of the shape.

### Quality reports

The post-generation quality check writes `quality_report.json` for programmatic analysis and `quality_report.md` for review. The reports contain structural validation results, warnings, exact and average-hash duplicate groups, exact duplicate groups spanning `train`, `val`, or `test`, per-class metadata summaries, and mask foreground-area summaries when masks are enabled.

An exact duplicate across subsets is reported as an error because it can create data leakage. Near-duplicate findings are warnings for manual review: an average hash is a screening method, not proof that two samples are semantically identical. Similarly, geometry and foreground-area statistics identify potentially problematic samples but do not replace task-specific scientific analysis.

### Segmentation masks

Use `--save-masks` in the CLI, or select **Save masks** in the GUI, to create a binary `L`-mode PNG mask for every image. Each mask uses `0` for background and `255` for the rendered shape, including its outline when the selected fill mode includes one. Masks are stored under `masks/<subset>/<class>/`; their relative paths are recorded in `metadata.csv` as `mask_filename`. This optional directory does not affect the ImageFolder-compatible `train/`, `val/`, and `test/` structure.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

For development and tests, install:

```powershell
pip install -r requirements-dev.txt
```

## Command-Line Usage

Generate a small dataset with default settings:

```powershell
python main.py
```

Generate a custom dataset:

```powershell
python main.py --output datasets/demo --train-count 100 --val-count 20 --test-count 20 --seed 42
```

Generate a dataset with one binary segmentation mask per image:

```powershell
python main.py --output datasets/segmentation-demo --train-count 100 --val-count 20 --test-count 20 --save-masks --seed 42
```

Generate a dataset using a total count and a standard train/val/test split:

```powershell
python main.py --output datasets/demo --dataset-count 100 --split-ratio 70/15/15 --seed 42
```

`--dataset-count` is the total number of images across all subsets. The supported automatic ratios are `70/15/15` and `80/10/10`. For manual mode, use `--train-count`, `--val-count`, and `--test-count`; these values are per class.

Useful CLI options:

```text
--config        Load generation settings from a generation_config.json file.
--save-config   Save the selected generation settings to a JSON file and exit.
--summary       Print a summary for an existing generated dataset and exit.
--analyze       Write an analysis report and figures for an existing generated dataset.
--estimate      Print an estimated output size before generation.
--output        Output dataset directory.
--image-size    Generated image width and height.
--train-count   Number of train images per class.
--val-count     Number of validation images per class.
--test-count    Number of test images per class.
--format        Image format: png, jpg, or gif.
--save-masks    Save binary PNG segmentation masks in masks/.
--color-mode    Image color mode: grayscale or rgb.
--background-color  RGB background color as three integers R G B, each from 0 to 255.
--foreground-color  RGB shape color as three integers R G B, each from 0 to 255.
--fill-mode     Shape fill mode: outline, filled, or outline_fill.
--rotation-min  Minimum rotation angle in degrees.
--rotation-max  Maximum rotation angle in degrees.
--scale-min     Minimum shape scale factor.
--scale-max     Maximum shape scale factor.
--center-offset-min  Minimum center offset in pixels.
--center-offset-max  Maximum center offset in pixels, up to half the image size.
--edge-clipping Allow shapes to be partially clipped by one image edge.
--min-visible-ratio  Minimum visible part of a clipped shape, from 0.6 to 1.0.
--noise-level   Uniform background noise level. Use 0 for a clean background.
--seed          Random seed.
--classes       Shape classes to generate.
```

RGB colors are entered as three separate values, not as a hexadecimal string. For example:

```powershell
python main.py --color-mode rgb --background-color 240 240 240 --foreground-color 20 80 200
```

### Configuration Files

Saved configuration files may include `subset_totals` when a total dataset size is distributed across the subsets. If present, this object must contain positive integer totals for all three keys: `train`, `val`, and `test`. The optional boolean `save_masks` enables binary PNG mask generation and defaults to `false` when absent, preserving compatibility with earlier configuration files. The GUI preserves exact subset totals when loading a configuration, including totals that cannot be divided evenly among the selected classes.

Every newly saved configuration includes `config_version`. Earlier valid configuration files without this field remain supported and are interpreted using the current compatibility version. Each generated dataset also contains `dataset_manifest.json`, which records SHA-256 hashes and byte sizes of the generated files. Use it to verify that a dataset used for an experiment has not been modified after generation.

During generation, the program rejects an encoded image that exactly duplicates an earlier image in the same dataset and deterministically retries with a derived sample seed. If the configured variability cannot produce a unique image after 100 attempts, generation stops with an error rather than creating a dataset with leakage. After generation, the program runs a quality check and prints the result. The check verifies the expected folder and file counts, image dimensions, actual encoded image format, color mode, metadata, class mapping, and required service files. It also identifies exact and near-duplicate images, detects exact duplicates across train/val/test, summarizes metadata geometry, and checks bounds that are outside the frame without edge clipping. When masks are enabled, it also verifies their directory structure, count, PNG format, `L` mode, dimensions, binary pixel values, metadata paths, and foreground-area statistics. The check writes `quality_report.json` and `quality_report.md` to the dataset directory. GIF files are verified as Pillow palettized (`P`) images because GIF serialization uses a palette.

### Controlled benchmark configurations

The [`experiments/configs`](experiments/configs) directory provides reproducible `clean`, `rotation`, `geometry`, `noise`, `clipping`, and `composite` dataset domains for controlled robustness experiments. See [experiments/README.md](experiments/README.md) for the protocol and run a domain with `python main.py --config experiments/configs/<name>.json`.

The same document describes the optional PyTorch training, evaluation, and result-aggregation pipeline. Its dependencies are deliberately separated in `requirements-experiments.txt` so they do not affect normal CLI or GUI use.

## Dataset analysis reports

Install the optional analysis dependency and analyse an existing generated dataset with:

```powershell
pip install -r requirements-analysis.txt
python main.py --analyze datasets/benchmarks/composite
```

The command writes `analysis/analysis_report.json`, `analysis/analysis_report.md`, and PNG figures under `analysis/figures/`. The report includes subset/class counts, descriptive statistics for sampled geometry and rendering parameters, optional mask foreground-area statistics, manifest and quality-report status, and warnings where class-level parameter means may be potential confounders. The figures include parameter distributions, a representative class montage, and examples with extreme scale, rotation, or area values.

Potential-confounder warnings are diagnostics for review, not statistical tests of significance or evidence of a causal effect. Before interpreting an analysis report, verify that `quality_report.json` passed and that `dataset_manifest.json` is valid. Archive the analysis report and figures together with the generation configuration, dependency file, Git revision, and dataset manifest used for the experiment.

## Reproducible environments

Runtime, development, and experiment dependencies are version-pinned in `requirements.txt`, `requirements-dev.txt`, and `requirements-experiments.txt`. The repository also provides `pyproject.toml`, so the application can be installed with `pip install .` and launched with `shape-domain-lab`. Record the Git revision, the relevant dependency file, the dataset configuration, and `dataset_manifest.json` with every reported experiment.

For the analysis tool, install `pip install .[analysis]` or use `requirements-analysis.txt`.

## Existing Output Directory

Before generation, ShapeDomainLab checks the output directory:

- An empty or non-existing directory is used automatically.
- If the directory contains a generated dataset, the application asks for confirmation before clearing it.
- If confirmation is given, all files inside the existing dataset directory are removed and a new dataset is generated.
- If confirmation is declined, generation is cancelled.
- A non-empty directory that is not a generated dataset is rejected. Its files are never removed.
- The same behavior is supported in both CLI and GUI modes.

## GUI Usage

Start the graphical interface:

```powershell
python -m src.gui.app
```

The GUI includes a menu bar, card-style parameter panels, a preview panel, summary and log panels, and a bottom status and generation control bar. It allows selecting dataset parameters, classes, rendering ranges, color mode, fill mode, rotation range, scale range, center offset range, optional edge clipping, optional background noise, image size, image format, output directory, and starts generation in a background worker. The **Save masks** checkbox is located on the same row as **Seed** and enables a binary PNG mask for each generated image. In grayscale mode, the Appearance section shows background and foreground intensity ranges. In RGB mode, the same area switches to background and shape color buttons that open a standard color picker. Edge clipping allows a single shape to be partially clipped by one image edge while keeping at least 60% of the shape visible. Image size is selected from common values: `28`, `32`, `64`, `128`, `224`, and `256`. Input fields include hover tooltips with descriptions and valid ranges; the `Tooltips` checkbox can show or hide them.

The bottom action bar provides the main workflow buttons: `Generate`, `Preview`, `Summary`, `Estimate`, `Save Config`, and `Load Config`. `Preview` renders sample images from the current parameters without creating a full dataset. `Summary` shows a summary for the selected output directory, and `Estimate` estimates generated image count and output size. `Save Config` writes the current settings to JSON. `Load Config` validates that the selected JSON file matches the ShapeDomainLab config format before applying it, preserves exact valid subset totals, and shows a highlighted error message if the file is not valid.

The summary and log panels show recent information in the main window and include `Open Summary` and `Open Log` buttons for reading the full text in larger separate windows. After generation, the GUI shows the quality check result and previews generated images from different classes. If the selected output directory already contains a generated dataset, the GUI loads preview images on startup or after browsing to that directory.

Use the `Clear Dataset` button to remove all files and folders inside the selected output directory. The application only clears directories that contain generated dataset structure with `train`, `val`, and `test` folders and image files, and asks for confirmation before clearing.

## Tests

Run automated tests:

```powershell
pytest
```

The tests generate small temporary datasets and validate the folder structure, metadata, class mapping, and quality checks.

The suite also checks byte-identical PNG and mask generation for a fixed seed, differing output for a changed seed, CLI generation and summary commands, headless GUI construction, and all version-controlled benchmark configurations. GitHub Actions runs the suite for Python 3.10, 3.11, and 3.12 on pull requests and changes to `main`.

## Notes for Git

Generated datasets are intentionally ignored by Git. The default output location is `datasets/shapes`. Keep source code, documentation, configuration examples, and tests in Git; keep generated images and large outputs outside Git.
