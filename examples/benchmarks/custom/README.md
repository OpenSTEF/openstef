<!--
SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Custom Benchmark Templates

Copy this folder as a starting point for your own BEAM benchmarks.

## Which file do I start with?

| I want to… | Start here |
|---|---|
| **Benchmark my own model** | `custom_forecaster.py` — implement `BacktestForecasterMixin` |
| **Benchmark on my own data** | `custom_benchmark.py` — extend `SimpleTargetProvider` |
| **Score predictions I already have** | `evaluate_existing_forecasts.py` |

## Files

| File | Role |
|---|---|
| `custom_forecaster.py` | **Template: your model.** Implements the `BacktestForecasterMixin` interface (config, quantiles, fit, predict). |
| `custom_benchmark.py` | **Template: your benchmark.** Defines where data lives, which metrics to use, and assembles the pipeline. |
| `run_liander2024_benchmark.py` | **Entry point:** test your forecaster on the built-in Liander 2024 dataset (auto-downloaded). |
| `run_custom_benchmark.py` | **Entry point:** run your forecaster on your own data (uses `custom_benchmark.py`). |
| `evaluate_existing_forecasts.py` | **Entry point:** bring your own prediction parquets, skip backtesting. |
| `compare_benchmark_runs.py` | **Entry point:** compare results from multiple runs side-by-side. |

## Quick start

```bash
# Install (requires uv: https://docs.astral.sh/uv/)
uv sync

# Test the example forecaster on Liander 2024
uv run python -m examples.benchmarks.custom.run_liander2024_benchmark

# Run with your custom data/targets
uv run python -m examples.benchmarks.custom.run_custom_benchmark
```

## Creating your own

### 1. Write a forecaster

Copy `custom_forecaster.py` and implement two methods:

- **`fit(data)`** — called periodically with recent history. Train your model here.
- **`predict(data)`** — called every few hours. Return a `TimeSeriesDataset` with a `"load"` column and one column per quantile (e.g. `"quantile_P05"`, `"quantile_P50"`).

The `data` argument is a `RestrictedHorizonVersionedTimeSeries` — it enforces no-lookahead by only exposing data available at `data.horizon`.

### 2. Define a benchmark (optional)

Copy `custom_benchmark.py` if you want to use **your own data**. Override `_get_measurements_path_for_target()` and `_get_weather_path_for_target()` to point to your parquet files.

If you're fine with the Liander 2024 dataset, skip this step and use `create_liander2024_benchmark_runner()` directly.

### 3. Run it

Copy `run_custom_benchmark.py`. Register your models as forecaster factories and call `pipeline.run()`.

## Evaluating pre-existing forecasts

If you already have predictions, place them in this layout:

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
from examples.benchmarks.custom.custom_benchmark import create_custom_benchmark_runner
from openstef_beam.benchmarking import LocalBenchmarkStorage
from pathlib import Path
runner = create_custom_benchmark_runner(storage=LocalBenchmarkStorage(base_path=Path('./tmp')))
for t in runner.target_provider.get_targets(['solar_park']):
    print(t.group_name, '/', t.name)
"
```

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

Then run:

```bash
uv run python -m examples.benchmarks.custom.evaluate_existing_forecasts
```

Results are written to `./benchmark_results/`. Each model gets its own subfolder with backtest predictions, evaluation scores, and analysis plots.

## Comparing results

After running at least two models, generate side-by-side comparison plots (global, per-group, per-target). The scripts automatically detect which targets are available in all runs:

```bash
uv run python -m examples.benchmarks.custom.compare_benchmark_runs
```

Output (HTML plots) is saved to `./benchmark_results_comparison/`.

```{toctree}
:maxdepth: 1
:hidden:

Implement a Custom Forecaster <custom_forecaster>
Configure a Custom Benchmark <custom_benchmark>
Run on Liander 2024 Data <run_liander2024_benchmark>
Run on Your Own Data <run_custom_benchmark>
Evaluate Existing Forecasts <evaluate_existing_forecasts>
Compare Multiple Runs <compare_benchmark_runs>
```
