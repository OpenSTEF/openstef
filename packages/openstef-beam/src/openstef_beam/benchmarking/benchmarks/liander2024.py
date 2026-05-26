# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Liander 2024 Short Term Energy Forecasting Benchmark Setup."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, override

from huggingface_hub import snapshot_download  # type: ignore[reportUnknownVariableType]
from pydantic import Field

from openstef_beam.analysis import AnalysisConfig
from openstef_beam.analysis.visualizations import WindowedMetricVisualization
from openstef_beam.analysis.visualizations.grouped_target_metric_visualization import GroupedTargetMetricVisualization
from openstef_beam.analysis.visualizations.precision_recall_curve_visualization import PrecisionRecallCurveVisualization
from openstef_beam.analysis.visualizations.quantile_probability_visualization import QuantileProbabilityVisualization
from openstef_beam.analysis.visualizations.summary_table_visualization import SummaryTableVisualization
from openstef_beam.analysis.visualizations.timeseries_visualization import TimeSeriesVisualization
from openstef_beam.backtesting import BacktestConfig
from openstef_beam.benchmarking.benchmark_pipeline import BenchmarkPipeline
from openstef_beam.benchmarking.callbacks.base import BenchmarkCallback
from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
from openstef_beam.benchmarking.storage.base import BenchmarkStorage
from openstef_beam.benchmarking.target_provider import SimpleTargetProvider
from openstef_beam.evaluation import EvaluationConfig, Window
from openstef_beam.evaluation.metric_providers import (
    MetricProvider,
    PeakMetricProvider,
    RCRPSProvider,
    RCRPSSampleWeightedProvider,
    RMAEProvider,
)
from openstef_core.types import AvailableAt, Quantile

type Liander2024Category = Literal["mv_feeder", "station_installation", "transformer", "solar_park", "wind_park"]


class Liander2024TargetProvider(SimpleTargetProvider[BenchmarkTarget, list[Liander2024Category]]):
    """Target provider for Liander 2024 STEF benchmark."""

    measurements_path_template: str = Field(default="{name}.parquet", init=False)
    weather_path_template: str = Field(default="{name}.parquet", init=False)
    profiles_path: str = Field(default="profiles.parquet", init=False)
    prices_path: str = Field(default="EPEX.parquet", init=False)
    targets_file_path: str = Field(default="liander2024_targets.yaml", init=False)

    data_start: datetime = Field(
        default=datetime.fromisoformat("2024-02-01T00:00:00Z"),
        init=False,
        frozen=True,
        description=(
            "Earliest timestamp to consider for training and benchmarking. "
            "Defaults to 2024-02-01, because weather data before this date is incomplete."
        ),
    )

    @override
    def get_targets(self, filter_args: list[Liander2024Category] | None = None) -> list[BenchmarkTarget]:
        targets = super().get_targets(filter_args)
        if filter_args is not None:
            targets = [t for t in targets if t.group_name in filter_args]

        for target in targets:
            target.train_start = max(target.train_start, self.data_start)
            target.benchmark_start = max(target.benchmark_start, self.data_start)

        return targets

    @override
    def get_metrics_for_target(self, target: BenchmarkTarget) -> list[MetricProvider]:
        return [
            RMAEProvider(quantiles=[Quantile(0.5)], lower_quantile=0.01, upper_quantile=0.99),
            RCRPSProvider(lower_quantile=0.01, upper_quantile=0.99),
            PeakMetricProvider(
                limit_pos=target.upper_limit if target.upper_limit is not None else 0.0,
                limit_neg=target.lower_limit if target.lower_limit is not None else 0.0,
                beta=2,
            ),
            RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99),
        ]

    @override
    def _get_measurements_path_for_target(self, target: BenchmarkTarget) -> Path:
        return (
            self.data_dir
            / "load_measurements"
            / target.group_name
            / self.measurements_path_template.format(name=target.name)
        )

    @override
    def _get_weather_path_for_target(self, target: BenchmarkTarget) -> Path:
        return (
            self.data_dir
            / "weather_forecasts_versioned"
            / target.group_name
            / self.weather_path_template.format(name=target.name)
        )


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
        WindowedMetricVisualization(
            name="rCRPS_sample_weighted_windowed_7D",
            metric="rCRPS_sample_weighted",
            window=Window(lag=timedelta(hours=0), size=timedelta(days=7)),
        ),
        WindowedMetricVisualization(
            name="rCRPS_sample_weighted_windowed_21D",
            metric="rCRPS_sample_weighted",
            window=Window(lag=timedelta(hours=0), size=timedelta(days=21)),
        ),
        WindowedMetricVisualization(
            name="rCRPS_sample_weighted_windowed_30D",
            metric="rCRPS_sample_weighted",
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
        GroupedTargetMetricVisualization(
            name="rCRPS_sample_weighted_grouped",
            metric="rCRPS_sample_weighted",
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
    data_dir: Path | None = None,
    storage: BenchmarkStorage | None = None,
    callbacks: list[BenchmarkCallback] | None = None,
    target_provider: Liander2024TargetProvider | None = None,
) -> BenchmarkPipeline[BenchmarkTarget, list[Liander2024Category]]:
    """Create benchmark pipeline for Liander2024 dataset.

    Args:
        data_dir: Dataset directory. Downloads from HuggingFace if None.
        storage: Storage backend for results.
        callbacks: Callbacks to use during benchmarking.
        target_provider: Custom target provider. Creates default if None.

    Returns:
        Configured benchmark pipeline.

    Example:
        >>> from pathlib import Path
        >>> runner = create_liander2024_benchmark_runner(
        ...     data_dir=Path("./liander2024_dataset")
        ... )
    """
    if data_dir is None:
        data_dir = Path(
            snapshot_download(
                repo_id="OpenSTEF/liander2024-stef-benchmark",
                repo_type="dataset",
            )
        )

    return BenchmarkPipeline[BenchmarkTarget, list[Liander2024Category]](
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
        target_provider=target_provider or Liander2024TargetProvider(data_dir=data_dir),
        storage=storage,
        callbacks=callbacks,
    )
