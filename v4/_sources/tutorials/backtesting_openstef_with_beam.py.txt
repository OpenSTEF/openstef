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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 📊 Backtesting OpenSTEF Models with OpenSTEF-BEAM
#
# This tutorial demonstrates how to use **OpenSTEF-BEAM** (Backtesting, Evaluation, Analysis, Metrics) to systematically evaluate forecasting models. You'll learn how to:
#
# 1. **Configure benchmark experiments** with multiple model types
# 2. **Run parallel backtests** across dozens of energy assets
# 3. **Compare model performance** with standardized metrics
# 4. **Generate analysis reports** with interactive visualizations
#
# > **BEAM** provides a rigorous framework for model evaluation, ensuring fair comparisons and reproducible results.

# %% [markdown]
# ## 🔧 Environment Setup
#
# First, we configure thread settings to prevent conflicts with XGBoost's internal parallelization when running multiple processes.

# %%
# --- Thread Configuration ---
# Prevent thread contention when running parallel backtests with XGBoost
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- Standard Imports ---
import logging
import multiprocessing
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# %% [markdown]
# ## ⚙️ Benchmark Configuration
#
# Configure the benchmark parameters:
# - **Output paths** — where to store results for each model
# - **Forecast horizons** — how far ahead to predict (using ISO 8601 duration format)
# - **Quantiles** — prediction intervals for probabilistic evaluation

# %%
# Import types for configuration
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024Category
from openstef_core.types import LeadTime, Q  # LeadTime: forecast horizon, Q: quantile

# --- Output Paths ---
OUTPUT_PATH = Path("./benchmark_results")
BENCHMARK_RESULTS_PATH_XGBOOST = OUTPUT_PATH / "XGBoost"
BENCHMARK_RESULTS_PATH_GBLINEAR = OUTPUT_PATH / "GBLinear"

# --- Parallelization ---
N_PROCESSES = multiprocessing.cpu_count()  # Use all available CPU cores
print(f"🖥️  Running with {N_PROCESSES} parallel processes")

# --- Forecast Configuration ---
FORECAST_HORIZONS = [LeadTime.from_string("P3D")]  # 3-day ahead forecast (ISO 8601: P3D)

# Quantiles for probabilistic forecasting (7 quantiles covering 5th to 95th percentile)
PREDICTION_QUANTILES = [
    Q(0.05),
    Q(0.1),
    Q(0.3),  # Lower quantiles
    Q(0.5),  # Median
    Q(0.7),
    Q(0.9),
    Q(0.95),  # Upper quantiles
]

# --- Benchmark Filter (optional) ---
# Set to None to run all categories, or specify categories like:
# BENCHMARK_FILTER = [Liander2024Category.TRANSFORMER, Liander2024Category.MV_FEEDER]
BENCHMARK_FILTER: list[Liander2024Category] | None = None

# %% [markdown]
# ## 🛠️ Model Configuration
#
# We define a **common configuration** that both models share, then create model-specific variants. This ensures fair comparison by keeping all settings identical except the model type.
#
# ### Available Models:
# - **XGBoost** — Gradient boosting trees (handles complex nonlinear patterns)
# - **GBLinear** — Gradient boosted linear model (better extrapolation, faster)

# %%
# Import workflow configuration
from openstef_models.presets import ForecastingWorkflowConfig

# Common configuration shared by all models
# This ensures fair comparison by keeping all settings identical
common_config = ForecastingWorkflowConfig(
    model_id="benchmark_model_",
    run_name=None,
    model="flatliner",  # Placeholder - will be overwritten per model
    # Forecast settings
    horizons=FORECAST_HORIZONS,
    quantiles=PREDICTION_QUANTILES,
    # Model reuse: reuse trained model for same target (speeds up backtesting)
    model_reuse_enable=True,
    mlflow_storage=None,  # Disable MLflow for this demo
    # Weather feature column mappings (match dataset column names)
    radiation_column="shortwave_radiation",
    wind_speed_column="wind_speed_80m",  # 80m wind speed for better wind park predictions
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    # Additional features
    energy_price_column="EPEX_NL",  # Day-ahead electricity price
    rolling_aggregate_features=["mean", "median", "max", "min"],  # Rolling window stats
    # Logging
    verbosity=0,  # Quiet mode for batch processing
)

# %%
# Create model-specific configurations by copying common config and updating model type
xgboost_config = common_config.model_copy(update={"model": "xgboost"})
gblinear_config = common_config.model_copy(update={"model": "gblinear"})

print("✅ Model configurations created:")
print(f"   - XGBoost: {xgboost_config.model}")
print(f"   - GBLinear: {gblinear_config.model}")

# %% [markdown]
# ## 💾 Storage Configuration
#
# **LocalBenchmarkStorage** manages the file structure for benchmark results:
# ```
# benchmark_results/
# ├── XGBoost/
# │   ├── backtest/      # Raw predictions
# │   ├── evaluation/    # Metrics per target
# │   └── analysis/      # Visualizations (HTML)
# └── GBLinear/
#     └── ...
# ```

# %%
# Initialize storage backends for each model
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage

storage_xgboost = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_XGBOOST)
storage_gblinear = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_GBLINEAR)

print(f"📁 XGBoost results: {BENCHMARK_RESULTS_PATH_XGBOOST}")
print(f"📁 GBLinear results: {BENCHMARK_RESULTS_PATH_GBLINEAR}")

# %% [markdown]
# ## 🚀 Run Backtests
#
# Now we run the **Liander 2024 Benchmark** — a comprehensive evaluation suite that:
# 1. Downloads the benchmark dataset from HuggingFace Hub (if needed)
# 2. Runs backtests across 5 asset categories (transformers, feeders, solar/wind parks)
# 3. Computes metrics and generates analysis visualizations
#
# ⚠️ **Note**: This may take several minutes depending on your hardware.

# %%
# Import benchmark components
from openstef_beam.benchmarking.baselines.openstef4 import create_openstef4_preset_backtest_forecaster
from openstef_beam.benchmarking.benchmarks.liander2024 import create_liander2024_benchmark_runner
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback

# --- Run XGBoost Benchmark ---
print("🌲 Running XGBoost benchmark...")
create_liander2024_benchmark_runner(
    storage=storage_xgboost,
    callbacks=[StrictExecutionCallback()],  # Fail fast on errors
).run(
    forecaster_factory=create_openstef4_preset_backtest_forecaster(
        workflow_config=xgboost_config,
    ),
    run_name="xgboost",
    n_processes=N_PROCESSES,
    filter_args=BENCHMARK_FILTER,
)
print("✅ XGBoost benchmark complete!")

# --- Run GBLinear Benchmark ---
print("\n📈 Running GBLinear benchmark...")
create_liander2024_benchmark_runner(
    storage=storage_gblinear,
    callbacks=[StrictExecutionCallback()],
).run(
    forecaster_factory=create_openstef4_preset_backtest_forecaster(
        workflow_config=gblinear_config,
    ),
    run_name="gblinear",
    n_processes=N_PROCESSES,
    filter_args=BENCHMARK_FILTER,
)
print("✅ GBLinear benchmark complete!")

# %% [markdown]
# ## 📊 Compare Model Performance
#
# The **BenchmarkComparisonPipeline** generates side-by-side analysis of multiple models:
# - Global metrics across all targets
# - Per-category breakdowns (transformers, feeders, etc.)
# - Time-windowed performance analysis

# %%
# Run model comparison analysis
from openstef_beam.benchmarking import BenchmarkComparisonPipeline
from openstef_beam.benchmarking.benchmarks.liander2024 import LIANDER2024_ANALYSIS_CONFIG

# Create comparison pipeline
target_provider = create_liander2024_benchmark_runner(
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
).target_provider

comparison_pipeline = BenchmarkComparisonPipeline(
    analysis_config=LIANDER2024_ANALYSIS_CONFIG,
    storage=LocalBenchmarkStorage(base_path=OUTPUT_PATH),
    target_provider=target_provider,
)

# Generate comparison reports
print("📊 Generating comparison analysis...")
comparison_pipeline.run(
    run_data={
        "xgboost": storage_xgboost,
        "gblinear": storage_gblinear,
    }
)
print("✅ Comparison analysis complete!")

# %% [markdown]
# ## 📈 View Analysis Results
#
# The benchmark generates interactive HTML visualizations. Let's open the most important ones:
#
# ### Key Metrics:
# - **rCRPS** (relative Continuous Ranked Probability Score) — measures probabilistic forecast accuracy
# - **rMAE** (relative Mean Absolute Error) — measures point forecast accuracy
# - Lower values = better performance

# %%
# Open key analysis plots in browser
# HTML visualizations are interactive and best viewed in a browser
import os
import webbrowser

# Base path for analysis results
analysis_base = os.path.abspath("./benchmark_results/analysis/D-1T06:00")

# Define key visualizations to open
visualizations = [
    ("rCRPS Grouped by Category", "rCRPS_grouped.html"),
    ("rCRPS Time-Windowed (7 days)", "rCRPS_windowed_7D.html"),
]

print("🌐 Opening analysis visualizations in browser...\n")
for name, filename in visualizations:
    filepath = os.path.join(analysis_base, filename)
    if Path(filepath).exists():
        print(f"   📊 {name}")
        webbrowser.open(f"file://{filepath}")
    else:
        print(f"   ⚠️  {name} not found at {filepath}")

# %% [markdown]
# ### 🔍 Explore Individual Target Results
#
# You can also view time series plots for individual targets. Let's look at a transformer forecast:

# %%
# List available target-specific visualizations
import glob

# Find all time series plots for individual targets
target_plots = glob.glob("./benchmark_results/XGBoost/analysis/*/*/time_series_plot*.html")

if target_plots:
    print("📊 Available target-specific time series plots:\n")
    for i, plot in enumerate(sorted(target_plots)[:5]):  # Show first 5
        parts = plot.split("/")
        category = parts[-3]  # e.g., "transformer"
        target = parts[-2]  # e.g., "OS Apeldoorn"
        print(f"   {i + 1}. {category}/{target}")

    # Open the first transformer plot as an example
    transformer_plots = [p for p in target_plots if "transformer" in p]
    if transformer_plots:
        example_plot = os.path.abspath(transformer_plots[0])
        print(f"\n🌐 Opening example: {transformer_plots[0]}")
        webbrowser.open(f"file://{example_plot}")
else:
    print("⚠️  No target-specific plots found. Run the benchmark first.")

# %% [markdown]
# ---
#
# ## 🎯 Summary
#
# In this tutorial, you learned how to:
#
# 1. ✅ **Configure benchmark experiments** with `ForecastingWorkflowConfig`
# 2. ✅ **Run parallel backtests** using the Liander 2024 benchmark
# 3. ✅ **Compare models** (XGBoost vs GBLinear) with `BenchmarkComparisonPipeline`
# 4. ✅ **Analyze results** with interactive HTML visualizations
#
# ### 📁 Output Structure
#
# ```
# benchmark_results/
# ├── XGBoost/
# │   ├── backtest/       # Raw predictions (parquet)
# │   ├── evaluation/     # Metrics per target
# │   └── analysis/       # HTML visualizations
# ├── GBLinear/
# │   └── ...
# └── analysis/           # Comparison analysis (both models)
#     └── D-1T06:00/
#         ├── rCRPS_grouped.html      # Probabilistic accuracy by category
#         ├── rMAE_grouped.html       # Point forecast accuracy
#         └── summary.html            # Overall summary
# ```
#
# ### 🚀 Next Steps
#
# - Experiment with different `FORECAST_HORIZONS` (e.g., `"PT6H"`, `"P7D"`)
# - Add more quantiles for higher resolution prediction intervals
# - Filter specific categories with `BENCHMARK_FILTER`
# - Integrate MLflow for experiment tracking
