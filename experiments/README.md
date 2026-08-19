# Controlled Benchmark Configurations

This directory defines six reproducible dataset domains for robustness experiments. Each configuration uses the same five classes, image size, grayscale appearance, class balance, PNG output, binary masks, subset sizes, background-intensity range (160–200), foreground-intensity range (0–50), and line width (1–3 pixels). A domain changes only the factor named in its filename, except for `composite`, which combines all factors. The shared appearance variation provides sufficient unique rendered samples while remaining constant across domains.

| Configuration | Controlled variation |
|---|---|
| `clean.json` | Centered, unrotated, unscaled, clean images. |
| `rotation.json` | Rotation from -45 to 45 degrees. |
| `geometry.json` | Scale from 0.7 to 1.3 and center offset from 0 to 12 pixels. |
| `noise.json` | Uniform background noise with level 20. |
| `clipping.json` | Edge clipping with at least 60% nominal visibility. |
| `composite.json` | Rotation, geometry, noise, and edge clipping combined. |

Each configuration generates 1,000 training, 200 validation, and 500 test images per class. The output path is relative to the project root and can be changed before generation. Use a configuration with:

```powershell
python main.py --config experiments/configs/clean.json
```

The differing `seed` values make each named benchmark dataset independently reproducible. For paired experiments that require identical base samples across domains, copy a configuration and set a common seed deliberately; record that choice in the experiment protocol.

## Training and evaluation pipeline

The optional PyTorch pipeline is isolated from the desktop application's runtime dependencies. Install it with:

```powershell
pip install -r requirements-experiments.txt
```

Train either `tiny_cnn` or `resnet18` on a generated dataset, retaining the selected model, hyperparameters, seed, validation metrics, predictions, and epoch history:

```powershell
python -m experiments.train --dataset datasets/benchmarks/clean --output runs/clean-seed-1 --model tiny_cnn --seed 1
```

Evaluate the saved model on a named test domain and then aggregate every run directory into CSV and Markdown tables:

```powershell
python -m experiments.evaluate --dataset datasets/benchmarks/rotation --run runs/clean-seed-1 --name rotation
python -m experiments.aggregate_results --runs runs --output results/benchmark-results
```

For a completed robustness matrix, create the inferential-ready descriptive summaries and macro-F1 heatmaps with:

```bash
python -m experiments.analyze_matrix --runs runs/robustness-matrix --output results/robustness-matrix/statistical-summary
```

The statistical summary uses only the held-out test-domain metrics (360 files for the 2 × 6 × 5 × 6 matrix), reports mean and sample standard deviation over the five seeds, and keeps validation metrics out of test-domain comparisons.

Run each training regime with multiple independent model seeds. The pipeline stores `model.pt`, `run_config.json`, `history.json`, `metrics-validation.json`, `metrics-<domain>.json`, and per-sample prediction CSV files. Do not tune hyperparameters using the `test/` subsets.

## Recommended article protocol

For a defensible robustness experiment, generate all benchmark domains from their version-controlled configurations before fitting a model. Train on `clean`, use only its `val/` subset to choose the architecture and hyperparameters, and freeze that choice. Then evaluate the selected checkpoint once on `clean/test` and once on the `test/` subset of each shifted domain. The `--name` argument should identify the evaluated domain so that the resulting metric and prediction files remain unambiguous.

Repeat every training condition with several independently specified model seeds; the dataset configuration and its seed should remain fixed within a comparison. Report the mean and standard deviation across model seeds for accuracy, balanced accuracy, and macro-F1. Include the per-class metrics and exported predictions when analysing failures or class-specific effects. Do not select hyperparameters, epochs, or checkpoints using any `test/` metric.

For each reported run, archive the following artifacts with the article or an associated data repository:

- the exact dataset configuration JSON files and generated `generation_config.json` files;
- the generated `metadata.csv` and `quality_report.json` files;
- `run_config.json`, which records model settings, seed, device, Python version, operating-system information, and PyTorch versions;
- `history.json`, validation metrics, test-domain metrics, and prediction CSV files; and
- the software revision identifier and the dependency files `requirements.txt`, `requirements-dev.txt`, and `requirements-experiments.txt`.

The dependency files are version-pinned. Install the appropriate file in a clean environment before recreating a reported run, and preserve the generated dataset's `dataset_manifest.json` to verify artifact integrity.

Before model training, create and archive the analysis report for each benchmark dataset with `python main.py --analyze <dataset>`. Its parameter distributions, class balance, montage, and confounder warnings provide evidence that the intended domain shift is controlled rather than an incidental rendering artifact.

Treat confounder warnings as screening diagnostics rather than significance tests. Confirm a passing `quality_report.json` and a valid `dataset_manifest.json` before analysing or training on a dataset, then archive those files with the analysis report, generation configuration, dependency specification, and Git revision.

## Resumable full matrix

Run the complete 6 × 6 transfer matrix for five model seeds with:

```powershell
python -m experiments.run_matrix --device cuda
```

The runner uses 50 epochs for `tiny_cnn` and 20 for `resnet18`, as fixed by the clean-domain validation pilot. It writes `runs/robustness-matrix/experiment_progress.json`, skips completed training and evaluation stages, and asks `experiments.train` to resume an interrupted training run from `training_checkpoint.pt`. Each epoch checkpoint records the model, optimizer, data-loader generator, random-number-generator states, best validation score, and history. Re-run the same command after an interruption to continue the matrix.

The pipeline requests deterministic cuDNN behavior and records the execution environment, but exact bitwise equality can still depend on the PyTorch release, operating system, hardware, and selected device. For the strongest replication claim, use the recorded environment and run the reported commands on CPU or on an equivalently configured accelerator.
