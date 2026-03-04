<!--
SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Custom Benchmark Example

End-to-end examples for running and customizing OpenSTEF **BEAM** (Backtesting, Evaluation, Analysis, Metrics) benchmarks.

## What is BEAM?

BEAM replays historical data day by day, trains your model, makes forecasts, and scores them -- all without data leakage. It works with any model that implements the `BacktestForecasterMixin` interface.

## Files

| File | What it does |
|---|---|
| `example_baseline.py` | **Start here.** A minimal forecaster that predicts the median of recent history. Shows the `BacktestForecasterMixin` interface (`config`, `quantiles`, `fit`, `predict`). |
| `example_benchmark.py` | Defines a custom benchmark: target provider (where data lives), metrics, and pipeline assembly. Extends `SimpleTargetProvider` directly -- adapt this when you have your own data layout. |
| `run_liander2024_benchmark.py` | Runs the example baseline + GBLinear on the built-in **Liander 2024** dataset (auto-downloaded from HuggingFace). Good starting point if you just want to try things out. |
| `run_benchmark.py` | Same as above but uses the custom benchmark pipeline from `example_benchmark.py`. |
| `evaluate_forecasts.py` | **Bring your own forecasts.** Injects pre-existing predictions into the pipeline and runs only evaluation + analysis (no backtesting). |

## Quick Start

```bash
# 1. Clone the repo
git clone git@github.com:OpenSTEF/openstef.git -b "release/v4.0.0"
cd openstef

# 2. Install all packages (requires uv: https://docs.astral.sh/uv/)
uv sync --all-extras --all-groups --all-packages
```

### Run the Liander 2024 benchmark

Uses the built-in Liander 2024 dataset (auto-downloaded from HuggingFace). Runs the example baseline and GBLinear on all target categories.

```bash
uv run python -m examples.benchmarks.custom_benchmark.run_liander2024_benchmark
```

### Run the custom benchmark

Uses the custom target provider from `example_benchmark.py` with your own pipeline config. Runs on `solar_park` targets by default.

```bash
uv run python -m examples.benchmarks.custom_benchmark.run_benchmark
```

### Evaluate pre-existing forecasts (no backtesting)

If you already have predictions from your own model or external system, you can skip backtesting entirely. Save your forecasts in the expected format and run only evaluation + analysis:

```bash
uv run python -m examples.benchmarks.custom_benchmark.evaluate_forecasts
```

See `evaluate_forecasts.py` for the required data format and the `format_predictions()` helper.

Results are written to `./benchmark_results/`. Each model gets its own subfolder with backtest predictions, evaluation scores, and analysis plots.

## Creating Your Own

### 1. Write a forecaster

Copy `example_baseline.py` and implement two methods:

- **`fit(data)`** -- called periodically with recent history. Train your model here.
- **`predict(data)`** -- called every few hours. Return a `TimeSeriesDataset` with a `"load"` column and one column per quantile (e.g. `"quantile_P05"`, `"quantile_P50"`).

The `data` argument is a `RestrictedHorizonVersionedTimeSeries` -- it enforces no-lookahead by only exposing data available at `data.horizon`. Use `data.get_window(start, end, available_before)` to retrieve slices.

### 2. Define a benchmark (optional)

Copy `example_benchmark.py` if you want to use **your own data**. The key class is `SimpleTargetProvider` -- override `_get_measurements_path_for_target()` and `_get_weather_path_for_target()` to point to your parquet files.

If you're fine with the Liander 2024 dataset, skip this step and use `create_liander2024_benchmark_runner()` directly.

### 3. Write a runner

Copy `run_benchmark.py`. Register your models as forecaster factories and call `pipeline.run()`.
