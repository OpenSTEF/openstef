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
| `evaluate_existing_forecasts.py` | **Bring your own forecasts.** Points the pipeline at pre-existing prediction parquets and runs only evaluation + analysis (no backtesting). |
| `compare_liander2024_results.py` | Compare results from multiple runs on the **Liander 2024** dataset. Auto-detects which targets are available in all runs. |
| `compare_custom_results.py` | Compare results from multiple runs on the **custom** benchmark. Same auto-detection as above. |

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

If you already have predictions from your own model or external system, you can skip backtesting entirely. Place your forecast parquets in the expected directory layout and run only evaluation + analysis.

#### Required directory layout

```
benchmark_results/MyForecasts/
└── backtest/
    └── <group_name>/                   # e.g. "solar_park"
        └── <target_name>/              # e.g. "Within 15 kilometers of Opmeer_normalized"
            └── predictions.parquet
```

`group_name` and `target_name` must match the values from your targets YAML. You can list them:

```bash
uv run python -c "
from examples.benchmarks.custom_benchmark.example_benchmark import create_custom_benchmark_runner
for t in create_custom_benchmark_runner().target_provider.get_targets(['solar_park']):
    print(t.group_name, '/', t.name)
"
```

#### Required parquet format

Each `predictions.parquet` must have:

| Column | Type | Description |
|---|---|---|
| *(index)* `timestamp` | `DatetimeIndex` | When each prediction is valid for. 15-min intervals, tz-naive UTC. |
| `available_at` | `datetime64` | When the prediction was generated (enables D-1 / lead-time filtering). |
| `quantile_P05` | `float` | 5th percentile prediction. |
| `quantile_P50` | `float` | Median prediction (**required**). |
| `quantile_P95` | `float` | 95th percentile prediction. |
| ... | `float` | One column per quantile, named with `Quantile(x).format()`. |

Example rows:

```
timestamp (index)      available_at          quantile_P05  quantile_P50  quantile_P95
2023-01-15 12:00:00    2023-01-14 06:00:00   0.5           1.2           2.0
2023-01-15 12:15:00    2023-01-14 06:00:00   0.6           1.3           2.1
```

#### Run

```bash
uv run python -m examples.benchmarks.custom_benchmark.evaluate_existing_forecasts
```

See `evaluate_existing_forecasts.py` for the full script.

Results are written to `./benchmark_results/`. Each model gets its own subfolder with backtest predictions, evaluation scores, and analysis plots.

### Compare results across runs

After running at least two models, generate side-by-side comparison plots (global, per-group, per-target). The scripts automatically detect which targets are available in all runs.

```bash
# Compare on the Liander 2024 dataset
uv run python -m examples.benchmarks.custom_benchmark.compare_liander2024_results

# Compare on the custom benchmark
uv run python -m examples.benchmarks.custom_benchmark.compare_custom_results
```

Comparison output (HTML plots) is saved to `./benchmark_results_comparison/`.

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
