<!--
SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Liander 2024

Pre-made benchmarks on the [Liander 2024 STEF benchmark dataset](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark)
— an open dataset of Dutch energy grid measurements (solar, wind, consumption).

**No code changes needed.** Pick a notebook below and run it. Data is
auto-downloaded from HuggingFace.

```bash
# Run the XGBoost + GBLinear benchmark
uv run python -m examples.benchmarks.liander2024.run_xgboost_gblinear_benchmark
```

For a comparison of isotonic and reference-aligned conformal quantile
calibration on a GBLinear run, use:

```bash
uv run python -m examples.benchmarks.liander2024.run_calibration_benchmark
```

The same benchmark is available as the
`run_calibration_benchmark.ipynb` notebook for interactive exploration.

This calibration benchmark keeps its calibration window separate from the
final holdout and writes `metrics.csv`, `metadata.json`, and `timeseries.png` to
`benchmark_results/calibration/`. It is intended for method comparison, not as
a production-scale model benchmark.

```{toctree}
:maxdepth: 1

XGBoost & GBLinear <run_xgboost_gblinear_benchmark>
Ensemble Models <run_ensemble_benchmark>
Calibration <run_calibration_benchmark>
Compare Results <compare_benchmark_runs>
```
