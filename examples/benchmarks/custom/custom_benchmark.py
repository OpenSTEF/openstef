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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Custom Benchmark Configuration
#
# Defines a complete benchmark: where your data lives, which metrics to compute,
# and how to assemble the pipeline.
#
# **User story:** *"I want to benchmark on my own data."*
#
# Copy this file and modify `MyTargetProvider` to point at your dataset.
# The pipeline configuration (`create_custom_benchmark_runner`) shows all the
# knobs: backtest schedule, evaluation windows, analysis visualizations.
#
# **See also:**
# - [TargetProvider](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.TargetProvider.html) — abstract interface
# - [SimpleTargetProvider](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.target_provider.SimpleTargetProvider.html) — file-based implementation (what we extend here)
# - [BenchmarkPipeline](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.BenchmarkPipeline.html) — the orchestrator
# - [EvaluationConfig](https://openstef.github.io/openstef/api/generated/openstef_beam.evaluation.EvaluationConfig.html) — how predictions are sliced and scored
# - [Custom Forecaster template](./custom_forecaster.ipynb) — implement your model here

# %% tags=["remove-cell"]
"""Example: custom benchmark with your own target provider.

Shows how to extend SimpleTargetProvider to load your own data and build a
benchmark pipeline. Uses the Liander 2024 dataset as example data source --
replace paths and logic with your own.

Expected directory layout (customize via path overrides)::

    data_dir/
    ├── targets.yaml                    # Target definitions
    ├── load_measurements/
    │   └── <group_name>/<name>.parquet # Measurements per target
    └── features/
        └── <group_name>/<name>.parquet # Features per target (weather, etc.)
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# %%

from datetime import timedelta
from pathlib import Path
from typing import Literal, override

from huggingface_hub import snapshot_download
from pydantic import Field

from openstef_beam.analysis import AnalysisConfig
from openstef_beam.analysis.visualizations import WindowedMetricVisualization
from openstef_beam.analysis.visualizations.grouped_target_metric_visualization import GroupedTargetMetricVisualization
from openstef_beam.analysis.visualizations.quantile_probability_visualization import QuantileProbabilityVisualization
from openstef_beam.analysis.visualizations.summary_table_visualization import SummaryTableVisualization
from openstef_beam.analysis.visualizations.timeseries_visualization import TimeSeriesVisualization
from openstef_beam.backtesting import BacktestConfig
from openstef_beam.benchmarking import BenchmarkPipeline, BenchmarkTarget, StrictExecutionCallback
from openstef_beam.benchmarking.storage.base import BenchmarkStorage
from openstef_beam.benchmarking.target_provider import SimpleTargetProvider
from openstef_beam.evaluation import EvaluationConfig, Window
from openstef_beam.evaluation.metric_providers import MetricProvider, RCRPSProvider, RMAEProvider
from openstef_core.types import AvailableAt, LeadTime, Quantile

# Define your own target categories for filtering (must match group_name in targets.yaml)
type MyCategory = Literal["solar_park", "wind_park"]

# %% [markdown]
# ## Target Provider
#
# The [`TargetProvider`](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.TargetProvider.html)
# tells BEAM where your data lives and which metrics to compute.
# Here we extend [`SimpleTargetProvider`](https://openstef.github.io/openstef/api/generated/openstef_beam.benchmarking.target_provider.SimpleTargetProvider.html)
# which handles file-based datasets with a targets YAML + parquet files.
#
# **CUSTOMIZE HERE:** Change path templates, category types, and metric selection.

# %%


class MyTargetProvider(SimpleTargetProvider[BenchmarkTarget, list[MyCategory]]):
    """Custom target provider -- extend SimpleTargetProvider to load your own data.

    Configure path templates and data flags, then override methods to customize
    target filtering, metrics, and file resolution.
    """

    # Path templates -- adapt to your directory structure
    # {name} is replaced with target.name from targets.yaml
    targets_file_path: str = Field(default="liander2024_targets.yaml", init=False)
    measurements_path_template: str = Field(default="{name}.parquet", init=False)
    weather_path_template: str = Field(default="{name}.parquet", init=False)

    # Disable shared profiles and prices -- only per-target features are used
    # Set to True if you have shared data files (profiles.parquet, prices.parquet)
    use_profiles: bool = False
    use_prices: bool = False

    @override
    def get_targets(self, filter_args: list[MyCategory] | None = None) -> list[BenchmarkTarget]:
        """Load targets and optionally filter by category.

        Returns:
            Filtered list of benchmark targets.
        """
        # super().get_targets() reads targets from the YAML file
        targets = super().get_targets(filter_args)
        # Keep only targets whose group_name matches one of the filter categories
        if filter_args is not None:
            targets = [t for t in targets if t.group_name in filter_args]
        return targets

    @override
    def get_metrics_for_target(self, target: BenchmarkTarget) -> list[MetricProvider]:
        """Define which metrics to compute per target.

        Returns:
            List of metric providers.
        """
        # rMAE: deterministic accuracy at the median (lower is better)
        # rCRPS: probabilistic accuracy across all quantiles (lower is better)
        return [
            RMAEProvider(quantiles=[Quantile(0.5)], lower_quantile=Quantile(0.01), upper_quantile=Quantile(0.99)),
            RCRPSProvider(lower_quantile=Quantile(0.01), upper_quantile=Quantile(0.99)),
        ]

    @override
    def _get_measurements_path_for_target(self, target: BenchmarkTarget) -> Path:
        """Resolve path to load measurement parquet.

        Liander 2024 uses: data_dir/load_measurements/<group>/<name>.parquet
        Change this to match your directory structure.

        Returns:
            Path to the measurement parquet file.
        """
        return self.data_dir / "load_measurements" / target.group_name / f"{target.name}.parquet"

    @override
    def _get_weather_path_for_target(self, target: BenchmarkTarget) -> Path:
        """Resolve path to features parquet (weather, etc.).

        Liander 2024 uses: data_dir/weather_forecasts_versioned/<group>/<name>.parquet
        Change this to match your directory structure.

        Returns:
            Path to the features parquet file.
        """
        return self.data_dir / "weather_forecasts_versioned" / target.group_name / f"{target.name}.parquet"


# %% [markdown]
# ## Analysis Configuration
#
# Choose which visualizations and summary tables BEAM generates after evaluation.
# Add or remove providers to customize the output report.

# %%
# --- Analysis config: which plots and tables to generate after evaluation ---
ANALYSIS_CONFIG = AnalysisConfig(
    visualization_providers=[
        TimeSeriesVisualization(name="time_series"),
        WindowedMetricVisualization(
            name="rMAE_7D",
            metric=("rMAE", Quantile(0.5)),
            window=Window(lag=timedelta(hours=0), size=timedelta(days=7)),
        ),
        WindowedMetricVisualization(
            name="rCRPS_30D",
            metric="rCRPS",
            window=Window(lag=timedelta(hours=0), size=timedelta(days=30)),
        ),
        GroupedTargetMetricVisualization(name="rMAE_grouped", metric="rMAE", quantile=Quantile(0.5)),
        GroupedTargetMetricVisualization(name="rCRPS_grouped", metric="rCRPS"),
        SummaryTableVisualization(name="summary"),
        QuantileProbabilityVisualization(name="quantile_probability"),
    ],
)


# %% [markdown]
# ## Pipeline Assembly
#
# Wire everything together: backtest schedule, evaluation config, analysis, and target provider.
# See [`BacktestConfig`](https://openstef.github.io/openstef/api/generated/openstef_beam.backtesting.BacktestConfig.html)
# and [`EvaluationConfig`](https://openstef.github.io/openstef/api/generated/openstef_beam.evaluation.EvaluationConfig.html)
# for all available options.
#
# **CUSTOMIZE HERE:** Adjust `predict_interval`, `train_interval`, evaluation windows, and lead times.

# %%


def create_custom_benchmark_runner(
    storage: BenchmarkStorage,
    data_dir: Path | None = None,
) -> BenchmarkPipeline[BenchmarkTarget, list[MyCategory]]:
    """Assemble a benchmark pipeline with the custom target provider.

    Args:
        storage: Where to save results.
        data_dir: Dataset path. Downloads Liander 2024 from HuggingFace if None.

    Returns:
        Ready-to-run benchmark pipeline.
    """
    if data_dir is None:
        data_dir = Path(snapshot_download(repo_id="OpenSTEF/liander2024-stef-benchmark", repo_type="dataset"))

    return BenchmarkPipeline[BenchmarkTarget, list[MyCategory]](
        # Backtest: how to replay history
        backtest_config=BacktestConfig(
            prediction_sample_interval=timedelta(minutes=15),  # Data resolution
            predict_interval=timedelta(hours=6),  # New forecast every 6 hours
            train_interval=timedelta(days=7),  # Retrain model every 7 days
        ),
        # Evaluation: how to slice and score the results
        evaluation_config=EvaluationConfig(
            available_ats=[AvailableAt.from_string("D-1T06:00")],  # Day-ahead forecast at 06:00
            lead_times=[
                LeadTime.from_string("P1D"),  # 1 day ahead
            ],  # Evaluate all lead times
            windows=[  # Rolling windows for metrics
                Window(lag=timedelta(hours=0), size=timedelta(days=7)),
                Window(lag=timedelta(hours=0), size=timedelta(days=30)),
            ],
        ),
        analysis_config=ANALYSIS_CONFIG,
        target_provider=MyTargetProvider(data_dir=data_dir),
        storage=storage,
        callbacks=[StrictExecutionCallback()],  # Fail fast on errors
    )
