# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Liander 2024 Short Term Energy Forecasting Benchmark Setup."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar, cast, override

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download  # type: ignore[reportUnknownVariableType]

from openstef_beam.analysis import AnalysisConfig
from openstef_beam.analysis.visualizations.grouped_target_metric_visualization import GroupedTargetMetricVisualization
from openstef_beam.analysis.visualizations.precision_recall_curve_visualization import PrecisionRecallCurveVisualization
from openstef_beam.analysis.visualizations.quantile_probability_visualization import QuantileProbabilityVisualization
from openstef_beam.analysis.visualizations.summary_table_visualization import SummaryTableVisualization
from openstef_beam.analysis.visualizations.timeseries_visualization import TimeSeriesVisualization
from openstef_beam.analysis.visualizations.windowed_metric_visualization import WindowedMetricVisualization
from openstef_beam.backtesting import BacktestConfig
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkPipeline
from openstef_beam.benchmarking.callbacks.base import BenchmarkCallback
from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
from openstef_beam.benchmarking.storage.base import BenchmarkStorage
from openstef_beam.benchmarking.target_provider import TargetProvider
from openstef_beam.evaluation import EvaluationConfig
from openstef_beam.evaluation.metric_providers import MetricProvider, PeakMetricProvider, RCRPSProvider, RMAEProvider
from openstef_beam.evaluation.models.window import Window
from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.types import AvailableAt, Quantile


class Liander2024TargetProvider(TargetProvider[BenchmarkTarget, None]):
    """Provider for Liander 2024 benchmark targets."""

    DATASET_REPO_ID: ClassVar[str] = "OpenSTEF/liander2024-stef-benchmark"

    targets: list[dict[str, str | float]]

    def __init__(self) -> None:
        """Initialize the benchmark target provider by loading target definitions from HuggingFace."""
        path: str = hf_hub_download(
            repo_id=self.DATASET_REPO_ID,
            filename="liander2024_targets.yaml",
            repo_type="dataset",
        )

        with Path(path).open(encoding="utf-8") as f:
            targets = yaml.safe_load(f)
        super().__init__(targets=targets)  # type: ignore

    @override
    def get_targets(self, filter_args: None = None) -> list[BenchmarkTarget]:
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
                limit=float(target["limit"])
                if target.get("limit") is not None
                else 0,  # Currently there is no limit defined yet in the dataset
                benchmark_start=datetime(2024, 3, 1, tzinfo=UTC),
                benchmark_end=datetime(2024, 12, 31, tzinfo=UTC),
                train_start=datetime(2024, 1, 1, tzinfo=UTC),
            )
            for target in self.targets
        ]

    @override
    def get_measurements_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        """Get measurement data for a target.

        Args:
            target: The benchmark target.

        Returns:
            Versioned time series dataset containing load measurements.
        """
        load = pd.read_parquet(  # type: ignore
            f"hf://datasets/{self.DATASET_REPO_ID}/load_measurements/{target.group_name}/{target.name}.parquet"
        )

        # Ensure consistent column naming between the solar/wind parks and the substation measurements
        if "load_normalized" in load.columns:
            load = load.rename(columns={"load_normalized": "load"})

        # Drop all nan values in the target
        load = load.dropna(subset=["load"])  # type: ignore

        return VersionedTimeSeriesDataset.from_dataframe(
            data=load,
            sample_interval=timedelta(minutes=15),
        )

    @override
    def get_predictors_for_target(self, target: BenchmarkTarget) -> VersionedTimeSeriesDataset:
        """Get predictor data for a target.

        Args:
            target: The benchmark target.

        Returns:
            Versioned time series dataset containing weather, EPEX spot prices, and profile data.
        """
        # Load predictors from dataset individually and combine
        weather = pd.read_parquet(  # type: ignore
            f"hf://datasets/{self.DATASET_REPO_ID}/weather_forecasts_versioned/{target.group_name}/{target.name}.parquet"
        )
        weather_dataset = VersionedTimeSeriesDataset.from_dataframe(
            data=weather,
            sample_interval=timedelta(minutes=15),
        )

        epex = pd.read_parquet(f"hf://datasets/{self.DATASET_REPO_ID}/EPEX.parquet")  # type: ignore
        epex_dataset = VersionedTimeSeriesDataset.from_dataframe(
            data=epex,
            sample_interval=timedelta(minutes=15),
        )

        profiles = pd.read_parquet(f"hf://datasets/{self.DATASET_REPO_ID}/profiles.parquet")  # type: ignore
        profiles_dataset = VersionedTimeSeriesDataset.from_dataframe(
            data=profiles,
            sample_interval=timedelta(minutes=15),
        )

        datasets = [
            weather_dataset,
            epex_dataset,
            profiles_dataset,
        ]
        return VersionedTimeSeriesDataset.concat(datasets, mode="inner")

    @override
    def get_metrics_for_target(self, target: BenchmarkTarget) -> list[MetricProvider]:
        """Get metrics for evaluating a target.

        Args:
            target: The benchmark target.

        Returns:
            List of metric providers for evaluation.
        """
        return [
            RMAEProvider(quantiles=[Quantile(0.5)], lower_quantile=0.01, upper_quantile=0.99),
            RCRPSProvider(lower_quantile=0.01, upper_quantile=0.99),
            PeakMetricProvider(limit_pos=abs(target.limit), limit_neg=-abs(target.limit), beta=2),
        ]

    @override
    def get_evaluation_mask_for_target(self, target: BenchmarkTarget) -> pd.DatetimeIndex | None:
        """Get evaluation mask for a target.

        In the future this can be used to exclude certain periods from evaluation like flatlines.

        Args:
            target: The benchmark target (unused, no mask applied).

        Returns:
            None (no evaluation mask applied).
        """
        return None


LIANDER2024_ANALYSIS_CONFIG = AnalysisConfig(
    visualization_providers=[
        TimeSeriesVisualization(name="time_series"),
        WindowedMetricVisualization(
            name="rMAE_windowed_7D",
            metric=("rMAE", Quantile(0.5)),
            window=Window(lag=timedelta(hours=0), size=timedelta(days=7)),
        ),
        WindowedMetricVisualization(
            name="rMAE_windowed_21D",
            metric=("rMAE", Quantile(0.5)),
            window=Window(lag=timedelta(hours=0), size=timedelta(days=21)),
        ),
        WindowedMetricVisualization(
            name="rMAE_windowed_30D",
            metric=("rMAE", Quantile(0.5)),
            window=Window(lag=timedelta(hours=0), size=timedelta(days=30)),
        ),
        WindowedMetricVisualization(
            name="rCRPS_windowed_7D",
            metric="rCRPS",
            window=Window(lag=timedelta(hours=0), size=timedelta(days=7)),
        ),
        WindowedMetricVisualization(
            name="rCRPS_windowed_21D",
            metric="rCRPS",
            window=Window(lag=timedelta(hours=0), size=timedelta(days=21)),
        ),
        WindowedMetricVisualization(
            name="rCRPS_windowed_30D",
            metric="rCRPS",
            window=Window(lag=timedelta(hours=0), size=timedelta(days=30)),
        ),
        GroupedTargetMetricVisualization(
            name="rMAE_grouped",
            metric="rMAE",
            quantile=Quantile(0.5),
        ),
        GroupedTargetMetricVisualization(
            name="rCRPS_grouped",
            metric="rCRPS",
        ),
        GroupedTargetMetricVisualization(name="best_f2", metric="effective_F2.0", selector_metric="effective_F2.0"),
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
)


def create_liander2024_benchmark_runner(
    storage: BenchmarkStorage | None = None,
    callbacks: list[BenchmarkCallback] | None = None,
) -> BenchmarkPipeline[BenchmarkTarget, None]:
    """Create a simple benchmark pipeline.

    Args:
        storage: Storage backend for benchmark results.
        callbacks: List of benchmark callbacks to use during benchmarking.

    Returns:
        Configured benchmark pipeline instance.
    """
    return BenchmarkPipeline[BenchmarkTarget, None](
        backtest_config=BacktestConfig(
            prediction_sample_interval=timedelta(minutes=15),
            predict_interval=timedelta(hours=6),
            train_interval=timedelta(days=7),
        ),
        evaluation_config=EvaluationConfig(
            available_ats=[AvailableAt.from_string("D-1T06:00")],
            lead_times=[],
            windows=[
                Window(lag=timedelta(hours=0), size=timedelta(days=7)),
                Window(lag=timedelta(hours=0), size=timedelta(days=21)),
                Window(lag=timedelta(hours=0), size=timedelta(days=30)),
            ],
        ),
        analysis_config=LIANDER2024_ANALYSIS_CONFIG,
        target_provider=Liander2024TargetProvider(),
        storage=storage,
        callbacks=callbacks,
    )
