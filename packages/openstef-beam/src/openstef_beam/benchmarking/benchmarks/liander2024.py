# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Liander 2024 Short Term Energy Forecasting Benchmark Setup."""

from datetime import timedelta
from pathlib import Path

from huggingface_hub import snapshot_download  # type: ignore[reportUnknownVariableType]

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
from openstef_beam.benchmarking.target_provider import SimpleTargetProvider
from openstef_beam.evaluation import EvaluationConfig
from openstef_beam.evaluation.metric_providers import MetricProvider, PeakMetricProvider, RCRPSProvider, RMAEProvider
from openstef_beam.evaluation.models.window import Window
from openstef_core.types import AvailableAt, Quantile

# TODO(#718): find alternative for using functions instead of lambdas in target provider


def _measurements_path_for_target(target: BenchmarkTarget) -> Path:
    """Build path for target measurements.

    Returns:
        Path: Path to the measurements parquet file
    """
    return Path("load_measurements") / target.group_name / f"{target.name}.parquet"


def _weather_path_for_target(target: BenchmarkTarget) -> Path:
    """Build path for target weather data.

    Returns:
        Path: Path to the weather forecasts parquet file
    """
    return Path("weather_forecasts_versioned") / target.group_name / f"{target.name}.parquet"


def _profiles_path() -> Path:
    """Build path for profiles data.

    Returns:
        Path: Path to the profiles parquet file
    """
    return Path("profiles.parquet")


def _prices_path() -> Path:
    """Build path for prices data.

    Returns:
        Path: Path to the EPEX prices parquet file
    """
    return Path("EPEX.parquet")


def _targets_file_path() -> Path:
    """Build path for targets file.

    Returns:
        Path: Path to the targets YAML file
    """
    return Path("liander2024_targets.yaml")


def _metrics_for_target(target: BenchmarkTarget) -> list[MetricProvider]:
    """Build metrics list for a target.

    Returns:
        list[MetricProvider]: List of metric providers for evaluation
    """
    return [
        RMAEProvider(quantiles=[Quantile(0.5)], lower_quantile=0.01, upper_quantile=0.99),
        RCRPSProvider(lower_quantile=0.01, upper_quantile=0.99),
        PeakMetricProvider(
            limit_pos=abs(target.upper_limit) if target.upper_limit is not None else 0.0,
            limit_neg=-abs(target.lower_limit) if target.lower_limit is not None else 0.0,
            beta=2,
        ),
    ]


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


def create_liander2024_target_provider(
    data_dir: Path | None = None,
) -> SimpleTargetProvider[BenchmarkTarget, None]:
    """Create target provider for Liander2024 dataset.

    Args:
        data_dir: Dataset directory. Downloads from HuggingFace if None.

    Returns:
        Configured target provider instance.
    """
    if data_dir is None:
        data_dir = Path(
            snapshot_download(
                repo_id="OpenSTEF/liander2024-stef-benchmark",
                repo_type="dataset",
            )
        )

    return SimpleTargetProvider(
        data_dir=Path(data_dir),
        measurements_path_for_target=_measurements_path_for_target,
        weather_path_for_target=_weather_path_for_target,
        profiles_path=_profiles_path,
        prices_path=_prices_path,
        targets_file_path=_targets_file_path,
        data_sample_interval=timedelta(minutes=15),
        metrics=_metrics_for_target,
        use_profiles=True,
        use_prices=True,
    )


def create_liander2024_benchmark_runner(
    data_dir: Path | None = None,
    storage: BenchmarkStorage | None = None,
    callbacks: list[BenchmarkCallback] | None = None,
) -> BenchmarkPipeline[BenchmarkTarget, None]:
    """Create benchmark pipeline for Liander2024 dataset.

    Args:
        data_dir: Dataset directory. Downloads from HuggingFace if None.
        storage: Storage backend for results.
        callbacks: Callbacks to use during benchmarking.

    Returns:
        Configured benchmark pipeline.

    Example:
        >>> from pathlib import Path
        >>> runner = create_liander2024_benchmark_runner(
        ...     data_dir=Path("./liander2024_dataset")
        ... )
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
        target_provider=create_liander2024_target_provider(data_dir),
        storage=storage,
        callbacks=callbacks,
    )
