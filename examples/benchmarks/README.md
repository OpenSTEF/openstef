<!--
SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Benchmarks

End-to-end benchmarking using **BEAM** (Backtesting, Evaluation, Analysis, Metrics).

BEAM replays historical data day by day, trains your model, makes forecasts, and scores them — all without data leakage.

## Which notebook do I need?

| I want to… | Start here |
|---|---|
| **See how OpenSTEF performs** (just run, no code changes) | [XGBoost & GBLinear](liander2024/run_xgboost_gblinear_benchmark) |
| **Benchmark my own model** | [Implement a Custom Forecaster](custom/custom_forecaster) |
| **Benchmark on my own data** | [Configure a Custom Benchmark](custom/custom_benchmark) |
| **Score predictions I already have** | [Evaluate Existing Forecasts](custom/evaluate_existing_forecasts) |

## Quick start

```bash
# Install (requires uv: https://docs.astral.sh/uv/)
uv sync

# Run the built-in Liander 2024 benchmark (XGBoost + GBLinear)
uv run python -m examples.benchmarks.liander2024.run_xgboost_gblinear_benchmark
```

## Liander 2024

Pre-made benchmarks on the [Liander 2024 STEF benchmark dataset](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark).
No code changes needed — just run.

## Build Your Own

Templates for benchmarking custom models or custom data. See the
[Build Your Own](custom/README) section for a detailed walkthrough.
