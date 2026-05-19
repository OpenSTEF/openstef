# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ensemble Model Benchmark
#
# Run an ensemble of multiple base models (e.g. LightGBM + GBLinear) with a learned
# weight combiner on the
# [Liander 2024 STEF benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark).
#
# **What this does:**
#
# 1. Downloads the Liander 2024 dataset from HuggingFace (automatic)
# 2. Trains multiple base models and a combiner that learns optimal weights
# 3. Produces probabilistic forecasts (7 quantiles) for a 36-hour horizon
# 4. Saves results locally for comparison
#
# **No code changes needed.** To benchmark your own model, see
# [Implement a Custom Forecaster](../custom/custom_forecaster.ipynb).
#
# ```{admonition} Ensemble types
# Change `ensemble_type` below to try different strategies:
# - `"learned_weights"` — a combiner model learns per-quantile weights
# - `"stacking"` — base model outputs become features for a meta-model
# - `"rules"` — fixed rule-based combination
# ```

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# %% [markdown]
# ## Setup

# %%
import logging
from datetime import timedelta
from pathlib import Path

from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig
from openstef_beam.benchmarking.baselines.openstef4 import (
    create_openstef4_preset_backtest_forecaster,
)
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024Category, create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_core.types import LeadTime, Q
from openstef_meta.presets import (
    EnsembleForecastingWorkflowConfig,
)
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.transforms.general import SampleWeightConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# %% [markdown]
# ## Ensemble configuration
#
# Choose which base models to combine and how. The `ensemble_type` controls the
# combination strategy; `base_models` lists which individual models to train.

# %%
OUTPUT_PATH = Path("./benchmark_results")

N_PROCESSES = int(os.environ.get("OPENSTEF_N_PROCESSES", "1"))

ensemble_type = "learned_weights"  # "stacking", "learned_weights" or "rules"
base_models = ["lgbm", "gblinear"]  # combination of "lgbm", "gblinear", "xgboost" and "lgbm_linear"
combiner_model = "lgbm"  # "lgbm", "xgboost", "rf" or "logistic" for learned weights; "gblinear" for stacking

model = "Ensemble_" + "_".join(base_models) + "_" + ensemble_type + "_" + combiner_model

# Forecast 36 hours ahead, producing 7 quantile bands
FORECAST_HORIZONS = [LeadTime.from_string("PT36H")]
PREDICTION_QUANTILES = [
    Q(0.05),
    Q(0.1),
    Q(0.3),
    Q(0.5),
    Q(0.7),
    Q(0.9),
    Q(0.95),
]

# Set to a list of categories to run only a subset (e.g. [Liander2024Category.SOLAR])
BENCHMARK_FILTER: list[Liander2024Category] | None = None

# %% [markdown]
# ## Workflow configuration
#
# `EnsembleForecastingWorkflowConfig` extends the standard config with ensemble-specific
# settings: which base models to use, the combiner strategy, and per-model sample weights.

# %%
USE_MLFLOW_STORAGE = os.environ.get("OPENSTEF_MLFLOW_STORAGE", "true").lower() == "true"

if USE_MLFLOW_STORAGE:
    storage = MLFlowStorage(
        tracking_uri=str(OUTPUT_PATH / "mlflow_artifacts"),
        local_artifacts_path=OUTPUT_PATH / "mlflow_tracking_artifacts",
    )
else:
    storage = None

workflow_config = EnsembleForecastingWorkflowConfig(
    model_id="common_model_",
    ensemble_type=ensemble_type,
    base_models=base_models,  # type: ignore
    combiner_model=combiner_model,
    horizons=FORECAST_HORIZONS,
    quantiles=PREDICTION_QUANTILES,
    model_reuse_enable=False,
    mlflow_storage=None,
    radiation_column="shortwave_radiation",
    rolling_aggregate_features=["mean", "median", "max", "min"],
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
    forecaster_sample_weights={
        "gblinear": SampleWeightConfig(method="exponential", weight_exponent=1.0),
        "lgbm": SampleWeightConfig(weight_exponent=0.0),
        "xgboost": SampleWeightConfig(weight_exponent=0.0),
        "lgbm_linear": SampleWeightConfig(weight_exponent=0.0),
    },
)

# %% [markdown]
# ## Backtest schedule
#
# The `BacktestForecasterConfig` controls how BEAM schedules training and prediction
# windows. Ensemble models typically need more context than single models.

# %%
backtest_config = BacktestForecasterConfig(
    requires_training=True,
    predict_length=timedelta(days=7),
    predict_min_length=timedelta(minutes=15),
    predict_context_length=timedelta(days=14),  # Context needed for lag features
    predict_context_min_coverage=0.5,
    training_context_length=timedelta(days=90),  # Three months of training data
    training_context_min_coverage=0.5,
    predict_sample_interval=timedelta(minutes=15),
)

# %% [markdown]
# ## Run the benchmark

# %%
if __name__ == "__main__":
    start_time = time.time()
    create_liander2024_benchmark_runner(
        storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH / model),
        data_dir=None,  # Path("../data/liander2024-energy-forecasting-benchmark"),
        callbacks=[StrictExecutionCallback()],
    ).run(
        forecaster_factory=create_openstef4_preset_backtest_forecaster(
            workflow_config=workflow_config,
            cache_dir=OUTPUT_PATH / "cache",
        ),
        run_name=model,
        n_processes=N_PROCESSES,
        filter_args=BENCHMARK_FILTER,
    )

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds.")
