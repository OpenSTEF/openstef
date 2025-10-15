"""Liander 2024 Benchmark Example.

====================================

This example demonstrates how to set up and run the Liander 2024 STEF benchmark using the OpenSTEF
benchmarking framework.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download  # type: ignore[reportUnknownVariableType]
from pydantic_extra_types.country import CountryAlpha2

from openstef_beam.analysis import AnalysisConfig
from openstef_beam.analysis.visualizations.grouped_target_metric_visualization import GroupedTargetMetricVisualization
from openstef_beam.analysis.visualizations.precision_recall_curve_visualization import PrecisionRecallCurveVisualization
from openstef_beam.analysis.visualizations.quantile_probability_visualization import QuantileProbabilityVisualization
from openstef_beam.analysis.visualizations.summary_table_visualization import SummaryTableVisualization
from openstef_beam.analysis.visualizations.timeseries_visualization import TimeSeriesVisualization
from openstef_beam.analysis.visualizations.windowed_metric_visualization import WindowedMetricVisualization
from openstef_beam.backtesting import BacktestConfig
from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkContext, BenchmarkPipeline
from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
from openstef_beam.benchmarking.storage import LocalBenchmarkStorage
from openstef_beam.benchmarking.target_provider import TargetProvider
from openstef_beam.evaluation import EvaluationConfig
from openstef_beam.evaluation.metric_providers import MetricProvider, RCRPSProvider, RMAEProvider
from openstef_beam.evaluation.models.window import Window
from openstef_core.datasets import ForecastDataset, TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.mixins import TransformPipeline
from openstef_core.types import AvailableAt, LeadTime, Q, Quantile
from openstef_models.models.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
    XGBoostForecasterConfig,
    XGBoostHyperParams,
)
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms import FeatureEngineeringPipeline
from openstef_models.transforms.general import ScalerTransform
from openstef_models.transforms.time_domain import HolidayFeaturesTransform
from openstef_models.transforms.time_domain.lag_transform import VersionedLagTransform

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# Benchmark execution settings
BENCHMARK_RUN_NAME = "liander_2024_xgboost"
BENCHMARK_RESULTS_PATH = Path("./benchmark_results")
N_PROCESSES = 8  # Number of parallel processes for benchmark execution

# HuggingFace dataset settings
DATASET_REPO_ID = "OpenSTEF/liander2024-stef-benchmark"
COUNTRY_CODE = CountryAlpha2("NL")

# Backtest configuration
PREDICTION_SAMPLE_INTERVAL = timedelta(minutes=15)
PREDICT_INTERVAL = timedelta(hours=6)  # How often predictions are made
TRAIN_INTERVAL = timedelta(days=7)  # How often model is retrained

# Model configuration
FORECAST_HORIZONS = [LeadTime.from_string("PT12H")]  # Forecast horizon(s)
PREDICTION_QUANTILES = [Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9)]  # Quantiles for probabilistic forecasts
LAG_FEATURES = [timedelta(days=-7)]  # Lag features to include
XGBOOST_HYPERPARAMS = XGBoostHyperParams()  # XGBoost hyperparameters
XGBOOST_VERBOSITY = 1

# Forecaster context settings
HORIZON_LENGTH = timedelta(hours=48)
HORIZON_MIN_LENGTH = timedelta(minutes=15)
PREDICT_CONTEXT_LENGTH = timedelta(days=14)
PREDICT_CONTEXT_MIN_COVERAGE = 0.5
TRAINING_CONTEXT_LENGTH = timedelta(days=30)
TRAINING_CONTEXT_MIN_COVERAGE = 0.5

# Evaluation configuration
EVALUATION_AVAILABLE_ATS = [AvailableAt.from_string("D-1T06:00")]  # Which forecasts to evaluate
EVALUATION_LEAD_TIMES: list[LeadTime] = []  # Specific lead times to evaluate
EVALUATION_WINDOWS: list[Window] = []  # Evaluation windows

# Analysis/Visualization windows
WINDOWED_METRIC_WINDOWS = [
    timedelta(days=7),
    timedelta(days=21),
    timedelta(days=30),
]


class BenchmarkTargetProvider(TargetProvider[BenchmarkTarget, None]):
    """Provider for Liander 2024 benchmark targets."""

    targets: list[dict[str, str | float]]

    def __init__(self) -> None:
        """Initialize the benchmark target provider by loading target definitions from HuggingFace."""
        # Load target definitions from HuggingFace dataset
        path = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename="liander2024_targets.yaml",
            repo_type="dataset",
        )

        with Path(path).open(encoding="utf-8") as f:
            targets = yaml.safe_load(f)
        super().__init__(targets=targets)  # type: ignore

    def get_targets(self, filter_args: None = None) -> list[BenchmarkTarget]:  # noqa: ARG002
        """Get all benchmark targets.

        Args:
            filter_args: Unused filter arguments.

        Returns:
            List of BenchmarkTarget instances.
        """
        # Create BenchmarkTarget instances from loaded target definitions
        return [
            BenchmarkTarget(
                name=cast(str, target["name"]),
                description=f"Energy load for {cast(str, target['group_name'])} {cast(str, target['name'])}",
                group_name=cast(str, target["group_name"]),
                latitude=cast(float, target["latitude"]),
                longitude=cast(float, target["longitude"]),
                limit=1500000,
                benchmark_start=datetime(2024, 3, 1, tzinfo=UTC),
                benchmark_end=datetime(2024, 12, 31, tzinfo=UTC),
                train_start=datetime(2024, 1, 1, tzinfo=UTC),
            )
            for target in self.targets
        ]

    def get_measurements_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:  # noqa: PLR6301
        """Get measurement data for a target.

        Args:
            target: The benchmark target.

        Returns:
            Versioned time series dataset containing load measurements.
        """
        # Load measured load data from dataset
        load = pd.read_parquet(  # type: ignore
            f"hf://datasets/{DATASET_REPO_ID}/load_measurements/{target.group_name}/{target.name}.parquet"
        )

        # Ensure consistent column naming between the solar/wind parks and the substation measurements
        if "load_normalized" in load.columns:
            load = load.rename(columns={"load_normalized": "load"})

        # Drop nan columns
        load = load.dropna(subset=["load"])  # type: ignore

        return VersionedTimeSeriesDataset.from_dataframe(
            data=load,
            sample_interval=PREDICTION_SAMPLE_INTERVAL,
        )

    def get_predictors_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:  # noqa: PLR6301
        """Get predictor data for a target.

        Args:
            target: The benchmark target.

        Returns:
            Versioned time series dataset containing weather, EPEX, and profile data.
        """
        # Load predictors from dataset individually and combine
        weather = pd.read_parquet(  # type: ignore
            f"hf://datasets/{DATASET_REPO_ID}/weather_forecasts_versioned/{target.group_name}/{target.name}.parquet"
        )
        weather_dataset = VersionedTimeSeriesDataset.from_dataframe(
            data=weather,
            sample_interval=PREDICTION_SAMPLE_INTERVAL,
        )

        epex = pd.read_parquet(f"hf://datasets/{DATASET_REPO_ID}/EPEX.parquet")  # type: ignore
        epex_dataset = VersionedTimeSeriesDataset.from_dataframe(
            data=epex,
            sample_interval=PREDICTION_SAMPLE_INTERVAL,
        )

        profiles = pd.read_parquet(f"hf://datasets/{DATASET_REPO_ID}/profiles.parquet")  # type: ignore
        profiles_dataset = VersionedTimeSeriesDataset.from_dataframe(
            data=profiles,
            sample_interval=PREDICTION_SAMPLE_INTERVAL,
        )

        datasets = [
            weather_dataset,
            epex_dataset,
            profiles_dataset,
        ]
        return VersionedTimeSeriesDataset.concat(datasets, mode="inner")

    def get_metrics_for_target(self, target: BenchmarkTarget) -> list[MetricProvider]:  # noqa: PLR6301, ARG002
        """Get metrics for evaluating a target.

        Args:
            target: The benchmark target (unused, same metrics for all targets).

        Returns:
            List of metric providers for evaluation.
        """
        return [RMAEProvider(), RCRPSProvider()]  # type: ignore

    def get_evaluation_mask_for_target(self, target: BenchmarkTarget) -> pd.DatetimeIndex | None:  # noqa: PLR6301, ARG002
        """Get evaluation mask for a target.

        Args:
            target: The benchmark target (unused, no mask applied).

        Returns:
            None (no evaluation mask applied).
        """
        # No evaluation mask
        return None


def create_benchmark_runner(
    storage: LocalBenchmarkStorage,
) -> BenchmarkPipeline[BenchmarkTarget, None]:
    """Create a simple benchmark pipeline.

    Args:
        storage: Storage backend for benchmark results.

    Returns:
        Configured benchmark pipeline instance.
    """
    # Create windowed metric visualizations based on configuration
    windowed_visualizations: list[WindowedMetricVisualization] = []
    for window_size in WINDOWED_METRIC_WINDOWS:
        days = int(window_size.total_seconds() / 86400)
        windowed_visualizations.extend([
            WindowedMetricVisualization(
                name=f"rMAE_windowed_{days}D",
                metric=("rMAE", Quantile(0.5)),
                window=Window(lag=timedelta(hours=0), size=window_size),
            ),
            WindowedMetricVisualization(
                name=f"rCRPS_windowed_{days}D",
                metric="rCRPS",
                window=Window(lag=timedelta(hours=0), size=window_size),
            ),
        ])

    return BenchmarkPipeline[BenchmarkTarget, None](
        backtest_config=BacktestConfig(
            prediction_sample_interval=PREDICTION_SAMPLE_INTERVAL,
            predict_interval=PREDICT_INTERVAL,
            train_interval=TRAIN_INTERVAL,
        ),
        evaluation_config=EvaluationConfig(
            available_ats=EVALUATION_AVAILABLE_ATS,
            lead_times=EVALUATION_LEAD_TIMES,
            windows=EVALUATION_WINDOWS,
        ),
        analysis_config=AnalysisConfig(
            visualization_providers=[
                TimeSeriesVisualization(name="time_series"),
                *windowed_visualizations,
                GroupedTargetMetricVisualization(
                    name="rMAE_grouped",
                    metric="rMAE",
                    quantile=Quantile(0.5),
                ),
                GroupedTargetMetricVisualization(
                    name="rCRPS_grouped",
                    metric="rCRPS",
                ),
                GroupedTargetMetricVisualization(
                    name="best_f2", metric="effective_F2.0", selector_metric="effective_F2.0"
                ),
                GroupedTargetMetricVisualization(
                    name="precision_at_best_f2", metric="effective_precision", selector_metric="effective_F2.0"
                ),
                GroupedTargetMetricVisualization(
                    name="recall_at_best_f2", metric="effective_recall", selector_metric="effective_F2.0"
                ),
                SummaryTableVisualization(
                    name="summary",
                ),
                PrecisionRecallCurveVisualization(
                    name="precision_recall_curve",
                    effective_precision_recall=False,
                ),
                PrecisionRecallCurveVisualization(
                    name="effective_precision_recall_curve",
                    effective_precision_recall=True,
                ),
                QuantileProbabilityVisualization(
                    name="quantile_probability",
                ),
            ]
        ),
        target_provider=BenchmarkTargetProvider(),
        storage=storage,
    )


class XGBoostBacktestForecaster(BacktestForecasterMixin):
    """Adapter for ForecastingModel to work with backtesting pipeline."""

    def __init__(self, config: BacktestForecasterConfig) -> None:
        """Initialize the XGBoost backtest forecaster.

        Args:
            config: Configuration for the backtest forecaster.
        """
        self.config = config
        self.model = ForecastingModel(
            preprocessing=FeatureEngineeringPipeline.create(
                horizons=FORECAST_HORIZONS,
                horizon_transforms=[
                    ScalerTransform(method="standard"),
                    HolidayFeaturesTransform(country_code=COUNTRY_CODE),
                ],
                versioned_transforms=[VersionedLagTransform(column="load", lags=LAG_FEATURES)],
            ),
            forecaster=XGBoostForecaster(
                config=XGBoostForecasterConfig(
                    horizons=FORECAST_HORIZONS,
                    quantiles=self.quantiles,
                    hyperparams=XGBOOST_HYPERPARAMS,
                    verbosity=XGBOOST_VERBOSITY,
                )
            ),
            postprocessing=TransformPipeline[ForecastDataset](transforms=[]),
            target_column="load",
        )

    @property
    def quantiles(self) -> list[Quantile]:
        """Return the list of quantiles for predictions."""
        return PREDICTION_QUANTILES

    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """Train the model on historical data.

        Args:
            data: Historical data for training.
        """
        self.model.fit(cast(VersionedTimeSeriesDataset, data.dataset))

    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Generate predictions for the forecast horizon.

        Args:
            data: Data for making predictions.

        Returns:
            Time series dataset with predictions, or None if prediction fails.
        """
        try:
            return self.model.predict(cast(VersionedTimeSeriesDataset, data.dataset))
        except Exception:  # noqa: BLE001
            return None


def forecaster_factory(
    context: BenchmarkContext,  # noqa: ARG001
    target: BenchmarkTarget,  # noqa: ARG001
) -> BacktestForecasterMixin:
    """Factory function to create forecaster for each target.

    Args:
        context: Benchmark execution context (unused).
        target: Target for which to create the forecaster (unused, same config for all).

    Returns:
        Configured forecaster ready for backtesting.
    """
    config = BacktestForecasterConfig(
        requires_training=True,
        horizon_length=HORIZON_LENGTH,
        horizon_min_length=HORIZON_MIN_LENGTH,
        predict_context_length=PREDICT_CONTEXT_LENGTH,
        predict_context_min_coverage=PREDICT_CONTEXT_MIN_COVERAGE,
        training_context_length=TRAINING_CONTEXT_LENGTH,
        training_context_min_coverage=TRAINING_CONTEXT_MIN_COVERAGE,
    )
    return XGBoostBacktestForecaster(config=config)


if __name__ == "__main__":
    storage = LocalBenchmarkStorage(base_path=BENCHMARK_RESULTS_PATH)
    runner = create_benchmark_runner(storage=storage)
    runner.run(
        forecaster_factory=forecaster_factory,
        run_name=BENCHMARK_RUN_NAME,
        n_processes=N_PROCESSES,
    )
