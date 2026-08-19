# Changelog

All notable changes to ShapeDomainLab are documented in this file.

## 1.0

### Changed

- Expanded `metadata.csv` to schema version `2.1`.
- Added generation provenance, nominal geometry, image-coordinate bounds, and clipping context to every metadata row.
- Added optional binary PNG segmentation masks and recorded their paths in `metadata.csv`.
- Added JSON and Markdown quality reports with duplicate screening, cross-subset leakage detection, geometry summaries, and optional mask statistics.
- Added six version-controlled benchmark configurations for controlled robustness experiments.
- Added an optional PyTorch experiment pipeline with reproducible training, evaluation, prediction export, and result aggregation.
- Added deterministic-generation and CLI end-to-end tests, plus a GitHub Actions test matrix for Python 3.10–3.12.
- Added configuration schema versions, SHA-256 dataset manifests, version-pinned dependencies, and package installation metadata for reproducible environments.
- Added optional dataset analysis reports with descriptive statistics, confounder warnings, montages, and parameter-distribution figures.
- Documented the metadata schema in the project README and generated dataset README.
