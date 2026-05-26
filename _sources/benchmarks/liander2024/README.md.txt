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

```{toctree}
:maxdepth: 1

XGBoost & GBLinear <run_xgboost_gblinear_benchmark>
Ensemble Models <run_ensemble_benchmark>
Compare Results <compare_benchmark_runs>
```
