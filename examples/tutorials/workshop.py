# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# %% tags=["remove-cell"]
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openstef_core.testing import configure_notebook_display, load_liander_dataset, setup_notebook_logging

configure_notebook_display()
logger = setup_notebook_logging(
    __name__,
    suppress=(
        "tqdm",
        "choreographer",
        "kaleido",
        "httpx",
        "huggingface_hub",
        "fsspec",
        "filelock",
        "openstef_core.datasets",
        "lightgbm",
    ),
)
logging.basicConfig(level=logging.WARNING, format="[%(asctime)s][%(levelname)s] %(message)s")

# %% [markdown]
# # ⚡ OpenSTEF Workshop: Forecasting Energy for Solar Parks
#
# This workshop follows one normalized solar park near Oosterwolde through a complete OpenSTEF workflow. We use the same target and forecast period while exploring the modular layers of OpenSTEF:
#
# 1. Explore real grid data: measurements, weather forecasts, profiles, and prices
# 2. Train and explain a default XGBoost model
# 3. Inspect the features created by preprocessing
# 4. Train GBLinear and learn when to trust each model with `openstef-meta`
# 5. Forecast with Chronos-2 through `openstef-foundation-models`
# 6. Compare the models with a short `openstef-beam` backtest
#
# The point is to compare model behavior, not only to produce a score.
#
# ### Questions to discuss
#
# - What would make a forecast useful to a distribution system operator beyond a low average error?
# - Which differences between the models do you expect for a solar park with self-curtailment?
# - Which parts of this workflow should be reusable when the target or model changes?

# %% [markdown]
# ## 🔧 Environment Setup
#
# We configure thread settings to keep the notebook responsive when model training and backtesting use parallel numerical libraries.
#
# ### Questions to discuss
#
# - Which parts of this cell are workshop plumbing, and which parts belong to the forecasting workflow?
# - Why does reproducible execution matter when people compare model outputs?

# %% [markdown]
# ## 💾 Download the Benchmark Dataset
#
# Download the files for one target through `load_liander_dataset`. The loader also fetches the shared profiles and prices, then caches everything in its default `./liander_dataset` directory.
#
# ### Questions to discuss
#
# - Which inputs are observations, which are forecasts, and which are metadata?
# - Why must a backtest use the forecast version available at the time?
# - What would data leakage look like in this workshop?

# %%
DATASET_DIR = Path("./liander_dataset")
TARGET_GROUP = "solar_park"
TARGET_NAME = "Within 15 kilometers of Oosterwolde_normalized"
TARGET_PATH = f"{TARGET_GROUP}/{TARGET_NAME}"

dataset = load_liander_dataset(target=TARGET_PATH, extra_files=["liander2024_targets.yaml"])
print(f"Dataset snapshot available at: {DATASET_DIR.resolve()}")

# %% [markdown]
# ## 🏗️ Meet Our Target: Solar Park near Oosterwolde
#
# The target provider reads the benchmark metadata and target-specific files from the downloaded snapshot. We select one normalized solar park so every later comparison uses the same target.
#
# ### Questions to discuss
#
# - What do the target's location, limits, and benchmark dates tell you about the problem?
# - Why might normalization help when comparing targets with different capacities?
# - Which operational events would you want the forecast to flag?

# %%
from openstef_beam.benchmarking.benchmarks.liander2024 import Liander2024TargetProvider

target_provider = Liander2024TargetProvider(data_dir=DATASET_DIR)
targets = target_provider.get_targets()
target = next(item for item in targets if item.name == TARGET_NAME and item.group_name == TARGET_GROUP)

print(f"Target: {target.name}")
print(f"Location: ({target.latitude:.3f}°N, {target.longitude:.3f}°E)")
print(f"Description: {target.description}")
print(f"Category: {target.group_name}")
print(f"Capacity limits: {target.lower_limit:.3f} MW to {target.upper_limit:.3f} MW")
print(f"Training starts: {target.train_start.date()}")
print(f"Benchmark period: {target.benchmark_start.date()} to {target.benchmark_end.date()}")

# %% [markdown]
# ## 📊 Explore the Input Data
#
# The target provider exposes target-specific versioned datasets. We keep the exploration separate from model construction so it is clear what each input contributes.
#
# ### Load Measurements
#
# Active power is measured every 15 minutes. Sharp dips can indicate self-curtailment events where production drops unexpectedly.
#
# ### Questions to discuss
#
# - What daily, seasonal, and event-driven patterns can you see?
# - Which deviations look predictable from the inputs?
# - Where would a forecast error matter most operationally?

# %%
load_dataset = target_provider.get_measurements_for_target(target)
load_df = load_dataset.select_version().data

print(f"Shape: {load_df.shape[0]:,} rows x {load_df.shape[1]} columns")
print(f"Period: {load_df.index.min()} to {load_df.index.max()}")
print(f"Columns: {list(load_df.columns)}")

fig = px.line(
    load_df,
    y="load",
    title=f"Active Power: {TARGET_NAME}",
    labels={"load": "Active Power (MW)", "index": "Time (UTC)"},
    template="plotly_white",
)
fig.add_hline(y=target.upper_limit, line_dash="dash", line_color="red", annotation_text="Upper capacity limit")
fig.add_hline(y=target.lower_limit, line_dash="dash", line_color="red", annotation_text="Lower capacity limit")
fig.update_layout(height=450, showlegend=False)
fig.show()

# %% [markdown]
# ### Weather Forecasts
#
# Solar radiation drives PV generation, while temperature and wind provide additional context. These are versioned weather forecasts with an `available_at` timestamp, which prevents look-ahead bias during backtesting.
#
# ### Questions to discuss
#
# - Which weather variable appears most connected to the solar signal?
# - Do the forecasts look smooth, noisy, or systematically delayed?
# - What would change if we used weather measurements instead of forecasts?

# %%
weather_dataset = target_provider.get_weather_for_target(target)
weather_df = weather_dataset.select_version().data

print(f"Shape: {weather_df.shape[0]:,} rows x {weather_df.shape[1]} columns")
print(f"Weather features: {list(weather_df.columns)}")

weather_columns = [
    ("temperature_2m", "Temperature (°C)", "royalblue"),
    ("shortwave_radiation", "Solar Radiation (W/m²)", "orange"),
    ("wind_speed_80m", "Wind Speed 80m (m/s)", "green"),
    ("surface_pressure", "Surface Pressure (hPa)", "purple"),
]
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[label for _, label, _ in weather_columns],
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)
for index, (column, label, color) in enumerate(weather_columns):
    row, column_index = divmod(index, 2)
    fig.add_trace(
        go.Scatter(
            x=weather_df.index,
            y=weather_df[column],
            name=label,
            line={"color": color, "width": 0.5},
            showlegend=False,
        ),
        row=row + 1,
        col=column_index + 1,
    )
fig.update_layout(title=f"Weather Forecasts: {TARGET_NAME}", height=600, template="plotly_white")
fig.show()

# %% [markdown]
# ### Standard Load Profiles
#
# Profiles capture recurring patterns and show how OpenSTEF can combine target-specific data with reusable contextual features.
#
# ### Questions to discuss
#
# - What repeating patterns are visible in the two-week window?
# - Why could a shared profile help when the target's own history is incomplete?
# - Which assumption about the relationship between profile and target should we test?

# %%
profiles_dataset = target_provider.get_profiles()
profiles_df = profiles_dataset.select_version().data.loc["2024-06-01":"2024-06-14"]
print(f"Profile columns: {list(profiles_df.columns)}")

fig = px.line(
    profiles_df,
    title="Standard Load Profiles",
    labels={"value": "Normalized Load", "index": "Time (UTC)", "variable": "Profile"},
    template="plotly_white",
)
fig.update_layout(height=400)
fig.show()

# %% [markdown]
# ### Day-Ahead Energy Prices
#
# When prices go negative, solar park operators may self-curtail to avoid paying to deliver energy. These events test whether a model can connect an external signal to an operational response.
#
# ### Questions to discuss
#
# - Can you identify negative-price periods?
# - Do price changes line up with load dips, or is the relationship conditional?
# - What other explanation could account for a drop in production?

# %%
prices_dataset = target_provider.get_prices()
prices_df = prices_dataset.select_version().data
print(f"Price columns: {list(prices_df.columns)}")

fig = px.line(
    prices_df,
    title="Day-Ahead Electricity Price (EPEX)",
    labels={"value": "Price (€/MWh)", "index": "Time (UTC)", "variable": ""},
    template="plotly_white",
)
fig.update_layout(height=400, showlegend=False)
fig.show()

# %% [markdown]
# ## 🛠️ Step 1: Train a Default XGBoost Model
#
# We now compose the aligned input datasets and choose one training and forecast period. XGBoost captures non-linear feature interactions well, but tree-based models do not extrapolate beyond values represented in training data.
#
# ### Questions to discuss
#
# - Which patterns should trees capture well in a solar park?
# - Where might a tree-based model struggle, especially near unseen peaks?
# - What evidence would distinguish a model limitation from a bad input feature?

# %%
from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow

combined_dataset = VersionedTimeSeriesDataset.concat(
    [load_dataset, weather_dataset, prices_dataset, profiles_dataset],
    mode="left",
).select_version()

TRAIN_START = datetime.fromisoformat("2024-02-01T00:00:00Z")
TRAIN_END = datetime.fromisoformat("2024-05-12T00:00:00Z")
FORECAST_START = TRAIN_END
FORECAST_END = FORECAST_START + timedelta(days=1)

train_dataset = combined_dataset.filter_by_range(start=TRAIN_START, end=TRAIN_END)
forecast_dataset = combined_dataset.filter_by_range(start=FORECAST_START, end=FORECAST_END)

QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]

print(f"Training: {TRAIN_START.date()} to {TRAIN_END.date()}")
print(f"Forecast: {FORECAST_START} to {FORECAST_END}")
print(f"Training features: {len(train_dataset.data.columns)} columns, {len(train_dataset.data):,} rows")

# %%
xgb_workflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id="demo_xgboost_default",
        model="xgboost",
        horizons=[LeadTime.from_string("PT24H")],
        quantiles=QUANTILES,
        target_column="load",
        radiation_column="shortwave_radiation",
        wind_speed_column="wind_speed_80m",
        pressure_column="surface_pressure",
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        energy_price_column="EPEX_NL",
        rolling_aggregate_features=["mean", "median", "max", "min"],
        mlflow_storage=None,
        verbosity=1,
    )
)

print("Training default XGBoost model...")
xgb_fit_result = xgb_workflow.fit(train_dataset)
print("Training complete.")

# %% [markdown]
# ### Forecast Preview: Default XGBoost
#
# The plot shows observed load, the median forecast, uncertainty bands, and target limits.
#
# ### Questions to discuss
#
# - Where does the forecast track the measured signal, and where does it miss?
# - Are the uncertainty bands wider at moments where you expect more risk?
# - How often does the forecast cross an operational limit?

# %%
from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_core.datasets import ForecastDataset

forecast_xgb_default: ForecastDataset = xgb_workflow.predict(
    forecast_dataset,
    forecast_start=FORECAST_START,
)
fig = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=forecast_dataset.data["load"])
    .add_model(
        model_name="XGBoost",
        forecast=forecast_xgb_default.median_series,
        quantiles=forecast_xgb_default.quantiles_data,
    )
    .add_limit(value=target.upper_limit, name="Upper limit")
    .add_limit(value=target.lower_limit, name="Lower limit")
    .plot(title=f"Default XGBoost: {TARGET_NAME}")
)
fig.update_layout(height=500, yaxis_title="Load (normalized)")
fig.show()

# %% [markdown]
# ### Feature Importances: Default XGBoost
#
# Feature importance connects a forecast back to signals the model used. Treat it as a clue about model behavior, not proof that a feature causes the target.
#
# ### Questions to discuss
#
# - Which features does the model find useful?
# - Do those features match your understanding of solar production and curtailment?
# - What test would you run before trusting a highly important feature?

# %%
from typing import cast

from openstef_models.explainability import ExplainableForecaster

explainable_xgb = cast(ExplainableForecaster, xgb_workflow.model.forecaster)
fig = explainable_xgb.plot_feature_importances()
fig.update_layout(title="Feature Importance: Default XGBoost", height=500)
fig.show()

# %% [markdown]
# ## 🔬 Inspect Derived Features
#
# OpenSTEF's preprocessing pipeline derives features from raw data. The fitted result makes this modular step visible: selectors and feature adders prepare the input, a forecaster produces predictions, and postprocessing orders the quantiles.
#
# ### Questions to discuss
#
# - Which columns are raw inputs and which were created by preprocessing?
# - Why do cyclic time features help represent daily or weekly behavior?
# - Which preprocessing step would you change first for another target type?

# %%
prepared_train = xgb_fit_result.input_data_train
derived_features = prepared_train.data
print(f"Derived features ({len(derived_features.columns)} total):")
print(list(derived_features.columns))

features_to_plot = [
    ("shortwave_radiation", "Shortwave Radiation (W/m²)"),
    ("dni", "DNI"),
    ("day_of_week_sine", "Day of Week (sine)"),
    ("time_of_day_sine", "Time of Day (sine)"),
]
window = derived_features.loc["2024-02-17":"2024-03-08"]
fig = make_subplots(
    rows=len(features_to_plot),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=[label for _, label in features_to_plot],
)
for row, (column, label) in enumerate(features_to_plot, start=1):
    if column in window.columns:
        fig.add_trace(
            go.Scatter(x=window.index, y=window[column], name=label, line={"width": 1}),
            row=row,
            col=1,
        )
fig.update_layout(height=500, template="plotly_white", showlegend=True)
fig.show()

# %% [markdown]
# ## 📈 Step 2: GBLinear and Modular Preprocessing
#
# XGBoost trees cannot extrapolate beyond the training range. **GBLinear** is a gradient-boosted linear model that can extrapolate linearly, giving us a useful contrast with the tree baseline.
#
# We keep sample weighting out of this first comparison so the model differences stay easy to interpret. The important workshop point is the shared workflow shape: a preset assembles preprocessing, a forecaster, and postprocessing from configuration.
#
# ### Questions to discuss
#
# - Which behavior should GBLinear capture that XGBoost may miss?
# - What does the fitted input reveal about work done before the model sees a row?
# - Which part of the workflow would you replace for custom preprocessing?

# %%
gbl_workflow = create_forecasting_workflow(
    config=ForecastingWorkflowConfig(
        model_id="demo_gblinear",
        model="gblinear",
        horizons=[LeadTime.from_string("PT24H")],
        quantiles=QUANTILES,
        target_column="load",
        radiation_column="shortwave_radiation",
        wind_speed_column="wind_speed_80m",
        pressure_column="surface_pressure",
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        energy_price_column="EPEX_NL",
        rolling_aggregate_features=["mean", "median", "max", "min"],
        mlflow_storage=None,
        verbosity=1,
    )
)

print("Training GBLinear model...")
gbl_fit_result = gbl_workflow.fit(train_dataset)
print("Training complete.")

# %% [markdown]
# ## 🧩 Step 3: Learn Which Model to Trust
#
# `openstef-meta` adds an ensemble workflow around the same forecasting components. The learned-weights combiner sees the base-model forecasts and learns which one to trust under different conditions.
#
# Here the ensemble uses XGBoost and GBLinear. They share the common workflow configuration, while each model keeps its own forecasting behavior and model-specific preprocessing.
#
# ### Questions to discuss
#
# - What does an ensemble add that simply averaging two forecasts would not?
# - When would you expect the ensemble to favor XGBoost, and when GBLinear?
# - What evidence would show that the combiner learned useful behavior instead of fitting noise?

# %%
from openstef_meta.presets import EnsembleForecastingWorkflowConfig, create_ensemble_forecasting_workflow

ensemble_config = EnsembleForecastingWorkflowConfig(
    model_id="demo_xgboost_gblinear_ensemble",
    ensemble_type="learned_weights",
    base_models=["xgboost", "gblinear"],
    combiner_model="lgbm",
    horizons=[LeadTime.from_string("PT24H")],
    quantiles=QUANTILES,
    target_column="load",
    radiation_column="shortwave_radiation",
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
    mlflow_storage=None,
)
ensemble_workflow = create_ensemble_forecasting_workflow(ensemble_config)
ensemble_fit_result = ensemble_workflow.fit(train_dataset)
ensemble_forecast: ForecastDataset = ensemble_workflow.predict(
    forecast_dataset,
    forecast_start=FORECAST_START,
)
print(f"Base models: {list(ensemble_config.base_models)}")
print(f"Combiner: {ensemble_config.combiner_model}")
print(f"Forecast rows: {len(ensemble_forecast.data):,}")

# %% [markdown]
# ## 🧠 Step 4: Forecast with Chronos-2
#
# `openstef-foundation-models` adds Chronos-2, a pretrained foundation model for zero-shot probabilistic forecasting. It does not train on this target. Instead, it receives recent load history and known-future covariates, then produces a forecast through the same workflow interface.
#
# We use the compact published checkpoint so the workshop stays practical on CPU. Chronos is kept as a separate model in this comparison because the current meta ensemble API accepts trainable base-model names only.
#
# ### Questions to discuss
#
# - What is different about zero-shot forecasting compared with fitting XGBoost or GBLinear here?
# - Which information can Chronos use after the forecast origin, and which would be leakage?
# - Does the Chronos forecast look like a local model or a broad prior over time-series behavior?

# %%
from openstef_foundation_models.models import Chronos2
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig as FoundationForecastingWorkflowConfig,
)
from openstef_foundation_models.presets.forecasting_workflow import (
    create_forecasting_workflow as create_foundation_forecasting_workflow,
)
from openstef_models.utils.feature_selection import Include

chronos_workflow = create_foundation_forecasting_workflow(
    FoundationForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=Chronos2.SMALL.checkpoint(),
        quantiles=QUANTILES,
        horizons=[LeadTime.from_string("PT24H")],
        target_column="load",
        selected_features=Include(
            "load",
            "shortwave_radiation",
            "wind_speed_80m",
            "temperature_2m",
        ),
        model_id="demo_chronos2",
    )
)
chronos_window = combined_dataset.filter_by_range(
    start=FORECAST_START - timedelta(days=60),
    end=FORECAST_END,
)
chronos_forecast: ForecastDataset = chronos_workflow.predict(
    chronos_window,
    forecast_start=FORECAST_START,
)
print(f"Chronos model fitted: {chronos_workflow.model.is_fitted}")
print(f"Forecast rows: {len(chronos_forecast.data):,}")

# %% [markdown]
# ## 🔍 Compare the Forecasts
#
# Compare XGBoost, GBLinear, their learned ensemble, and Chronos-2 at the same forecast origin. The models share the target and operational context, but they learn in different ways.
#
# ### Questions to discuss
#
# - Which model follows the actual signal most closely during this period?
# - Where do the models disagree, and what kind of uncertainty does that reveal?
# - Would you choose the best-looking median, the narrowest interval, or the most useful limit warnings?

# %%
comparison_window = combined_dataset.filter_by_range(
    start=FORECAST_START - timedelta(days=14),
    end=FORECAST_END,
)
xgb_comparison_forecast: ForecastDataset = xgb_workflow.predict(
    comparison_window,
    forecast_start=FORECAST_START,
)
gbl_comparison_forecast: ForecastDataset = gbl_workflow.predict(
    comparison_window,
    forecast_start=FORECAST_START,
)
comparison_actuals = combined_dataset.data["load"].loc[FORECAST_START:FORECAST_END]
comparison_plotter = (
    ForecastTimeSeriesPlotter()
    .add_measurements(measurements=comparison_actuals)
    .add_model(
        model_name="XGBoost",
        forecast=xgb_comparison_forecast.median_series,
        quantiles=xgb_comparison_forecast.quantiles_data,
    )
    .add_model(
        model_name="GBLinear",
        forecast=gbl_comparison_forecast.median_series,
        quantiles=gbl_comparison_forecast.quantiles_data,
    )
    .add_model(
        model_name="Ensemble",
        forecast=ensemble_forecast.median_series,
        quantiles=ensemble_forecast.quantiles_data,
    )
    .add_model(
        model_name="Chronos-2",
        forecast=chronos_forecast.median_series,
        quantiles=chronos_forecast.quantiles_data,
    )
)
fig = comparison_plotter.plot(title="XGBoost, GBLinear, Ensemble, and Chronos-2")
fig.update_layout(height=550, yaxis_title="Load (normalized)")
fig.show()

# %% [markdown]
# ## 🚀 Short Backtest with OpenSTEF BEAM
#
# One forecast plot builds intuition, but it does not tell us how a model behaves across repeated forecast origins. `openstef-beam` provides the backtesting and evaluation workflow. We keep the same target and shorten the benchmark period for a workshop run.
#
# Chronos is zero-shot, so it uses BEAM's foundation-model adapter instead of the train-and-predict adapter used by the other workflows.
#
# ### Questions to discuss
#
# - What does repeating the forecast reveal that one forecast cannot?
# - How would weekly retraining change the interpretation of the trainable-model results?
# - What should we check in the benchmark output before claiming one model is better?

# %%
from openstef_beam.benchmarking.baselines.openstef4 import create_openstef4_preset_backtest_forecaster
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext
from openstef_beam.benchmarking.benchmarks.liander2024 import (
    Liander2024Category,
    create_liander2024_benchmark_runner,
)
from openstef_beam.benchmarking.callbacks.strict_execution_callback import StrictExecutionCallback
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage.base import BenchmarkStorage
from openstef_beam.benchmarking.storage.local_storage import LocalBenchmarkStorage
from openstef_foundation_models.integrations.beam import FoundationModelBacktestForecaster

OUTPUT_PATH = Path("./benchmark_results")
BENCHMARK_RESULTS_PATH_XGBOOST = OUTPUT_PATH / "XGBoost"
BENCHMARK_RESULTS_PATH_GBLINEAR = OUTPUT_PATH / "GBLinear"
BENCHMARK_RESULTS_PATH_ENSEMBLE = OUTPUT_PATH / "Ensemble"
BENCHMARK_RESULTS_PATH_CHRONOS = OUTPUT_PATH / "Chronos-2"

common_config = ForecastingWorkflowConfig(
    model_id="benchmark_model_",
    run_name=None,
    model="flatliner",
    horizons=[LeadTime.from_string("PT24H")],
    quantiles=QUANTILES,
    model_reuse_enable=True,
    mlflow_storage=None,
    radiation_column="shortwave_radiation",
    wind_speed_column="wind_speed_80m",
    pressure_column="surface_pressure",
    temperature_column="temperature_2m",
    relative_humidity_column="relative_humidity_2m",
    energy_price_column="EPEX_NL",
    rolling_aggregate_features=["mean", "median", "max", "min"],
    verbosity=0,
)
xgboost_config = common_config.model_copy(update={"model": "xgboost"})
gblinear_config = common_config.model_copy(update={"model": "gblinear"})
benchmark_ensemble_config = ensemble_config.model_copy(update={"model_id": "benchmark_ensemble"})

storage_xgboost = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_XGBOOST)
storage_gblinear = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_GBLINEAR)
storage_ensemble = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_ENSEMBLE)
storage_chronos = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH_CHRONOS)

class ShortBenchmarkTargetProvider(Liander2024TargetProvider):
    """One target with a short benchmark period for a fast workshop run."""

    target_name: str
    target_group: str

    def get_targets(self, filter_args: list[Liander2024Category] | None = None) -> list[BenchmarkTarget]:
        """Return the selected target with a shortened benchmark period."""
        targets = super().get_targets(filter_args)
        targets = [item for item in targets if item.name == self.target_name and item.group_name == self.target_group]
        for item in targets:
            item.benchmark_start = datetime.fromisoformat("2024-05-01T00:00:00Z")
            item.benchmark_end = datetime.fromisoformat("2024-06-30T23:59:59Z")
        return targets

single_target_provider = ShortBenchmarkTargetProvider(
    data_dir=DATASET_DIR,
    target_name=TARGET_NAME,
    target_group=TARGET_GROUP,
)
targets = single_target_provider.get_targets()
assert len(targets) == 1, f"Expected 1 target, got {len(targets)}"
print(f"Benchmarking: {targets[0].name}")
print(f"Short benchmark period: {targets[0].benchmark_start.date()} to {targets[0].benchmark_end.date()}")

# %%
def run_benchmark(
    storage: BenchmarkStorage,
    workflow_config: ForecastingWorkflowConfig | EnsembleForecastingWorkflowConfig,
    run_name: str,
) -> None:
    """Run one trainable workflow through the short BEAM benchmark."""
    create_liander2024_benchmark_runner(
        storage=storage,
        callbacks=[StrictExecutionCallback()],
        target_provider=single_target_provider,
    ).run(
        forecaster_factory=create_openstef4_preset_backtest_forecaster(
            workflow_config=workflow_config,
        ),
        run_name=run_name,
        n_processes=1,
    )

run_benchmark(storage_xgboost, xgboost_config, "xgboost")
run_benchmark(storage_gblinear, gblinear_config, "gblinear")
run_benchmark(storage_ensemble, benchmark_ensemble_config, "ensemble")

# Chronos is zero-shot and reuses one loaded workflow across all forecast windows.
def create_chronos_benchmark_forecaster(
    _context: BenchmarkContext,
    _target: BenchmarkTarget,
) -> FoundationModelBacktestForecaster:
    return FoundationModelBacktestForecaster.from_workflow(
        chronos_workflow,
        predict_length=timedelta(days=1),
        predict_context_length=timedelta(days=60),
    )

create_liander2024_benchmark_runner(
    storage=storage_chronos,
    callbacks=[StrictExecutionCallback()],
    target_provider=single_target_provider,
).run(
    forecaster_factory=create_chronos_benchmark_forecaster,
    run_name="chronos2",
    n_processes=1,
)
print("All four backtests complete.")

# %% [markdown]
# ## 📊 Compare Backtest Results
#
# `BenchmarkComparisonPipeline` turns stored BEAM runs into standardized reports with time-series views, grouped metrics, summary tables, and operational limit analysis.
#
# ### Questions to discuss
#
# - In `summary.html`, which model has the lowest rMAE and rCRPS? Are they the same model?
# - In `rMAE_windowed_7D.html`, does the ranking stay stable through the benchmark period?
# - What do grouped and operational-limit reports show that one aggregate score hides?
# - Does the output structure make the result easy to reproduce and audit?

# %%
from openstef_beam.benchmarking import BenchmarkComparisonPipeline
from openstef_beam.benchmarking.benchmarks.liander2024 import LIANDER2024_ANALYSIS_CONFIG

COMPARISON_RESULT_PATH = OUTPUT_PATH / "comparison"
comparison_pipeline = BenchmarkComparisonPipeline(
    analysis_config=LIANDER2024_ANALYSIS_CONFIG,
    storage=LocalBenchmarkStorage(base_path=COMPARISON_RESULT_PATH),
    target_provider=single_target_provider,
)
comparison_pipeline.run(
    run_data={
        "xgboost": storage_xgboost,
        "gblinear": storage_gblinear,
        "ensemble": storage_ensemble,
        "chronos2": storage_chronos,
    }
)
print(f"Results saved to: {COMPARISON_RESULT_PATH.resolve()}")

# %% [markdown]
# ## 🎯 Workshop Summary
#
# This workshop followed one target through OpenSTEF's modular forecasting stack:
#
# 1. Explored target measurements, versioned weather, profiles, and prices
# 2. Trained and explained XGBoost
# 3. Inspected preprocessing and derived features
# 4. Trained GBLinear as a contrasting extrapolating model
# 5. Built an ensemble with `openstef-meta`
# 6. Forecast with Chronos-2 through `openstef-foundation-models`
# 7. Backtested four approaches with `openstef-beam`
#
# The conclusion should be supported by the forecast plot and the BEAM reports for this target and period.
#
# ### Questions to close with
#
# - Which model would you deploy for this target, and which evidence supports that choice?
# - What would you change first for a second target: data selection, preprocessing, model, or benchmark period?
# - Which OpenSTEF package owns each capability, and how does the modular setup make that boundary visible?
#
# ### 📁 Output Structure
#
# ```text
# benchmark_results/
# ├── XGBoost/
# ├── GBLinear/
# ├── Ensemble/
# ├── Chronos-2/
# └── comparison/
#     └── global/D-1T0600/
#         ├── rCRPS_windowed_7D.html
#         ├── rMAE_grouped.html
#         └── summary.html
# ```
