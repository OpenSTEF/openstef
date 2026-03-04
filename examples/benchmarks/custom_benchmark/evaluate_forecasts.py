"""Evaluate pre-existing forecasts without running backtesting.

If you already have forecast predictions (e.g. from your own model or an external
system), you can inject them into the benchmark pipeline and run only the evaluation
and analysis steps.

How it works:
1. Format your forecasts as a TimeSeriesDataset (see format_predictions() below)
2. Save them via LocalBenchmarkStorage.save_backtest_output() for each target
3. Run the benchmark pipeline -- it detects the existing backtest output and
   automatically skips to evaluation + analysis.

The pipeline still needs a "forecaster factory" to know which quantiles were used,
but fit() and predict() are never called.
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import logging
import multiprocessing
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from openstef_beam.benchmarking.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries

from examples.benchmarks.custom_benchmark.example_benchmark import create_custom_benchmark_runner
from openstef_beam.benchmarking import BenchmarkContext, BenchmarkTarget, LocalBenchmarkStorage
from openstef_beam.benchmarking.baselines import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import Q, Quantile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

OUTPUT_PATH = Path("./benchmark_results")
N_PROCESSES = multiprocessing.cpu_count()

# Quantiles your forecasts were generated for (must include 0.5 = median)
PREDICTION_QUANTILES = [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]


def format_predictions(
    timestamps: pd.DatetimeIndex,
    available_at: pd.Series,
    predictions: dict[float, list[float]],
    sample_interval: timedelta = timedelta(minutes=15),
) -> TimeSeriesDataset:
    """Format raw predictions into a TimeSeriesDataset that BEAM can evaluate.

    Args:
        timestamps: When each prediction is valid for (DatetimeIndex).
        available_at: When each prediction was made (datetime Series).
            This enables D-1 / lead-time filtering during evaluation.
        predictions: Mapping of quantile value → prediction values.
            E.g. {0.5: [1.0, 2.0, ...], 0.05: [0.5, 1.0, ...]}
        sample_interval: Time between consecutive timestamps.

    Returns:
        A TimeSeriesDataset ready for save_backtest_output().

    Example::

        ds = format_predictions(
            timestamps=pd.date_range("2023-01-01", periods=96, freq="15min"),
            available_at=pd.Series([datetime(2022, 12, 31, 6, 0)] * 96),
            predictions={0.5: median_values, 0.05: lower_values, 0.95: upper_values},
        )
    """
    # Build column dict: "available_at" + quantile columns (e.g. "quantile_P50")
    data = {"available_at": available_at}
    for q_value, values in predictions.items():
        col_name = Quantile(q_value).format()  # e.g. Q(0.05).format() -> "quantile_P05"
        data[col_name] = values

    return TimeSeriesDataset(
        data=pd.DataFrame(data, index=timestamps),
        sample_interval=sample_interval,
    )


class _QuantileStub(BacktestForecasterMixin):
    """Stub forecaster that only provides quantile info -- never actually runs.

    The benchmark pipeline needs a forecaster factory to know which quantiles
    were used. Since backtest output already exists, fit() and predict() are
    never called.
    """

    def __init__(self, quantiles: list[Quantile]) -> None:
        self._quantiles = quantiles
        self.config = BacktestForecasterConfig(
            requires_training=False,
            predict_length=timedelta(days=1),
            predict_context_length=timedelta(minutes=15),
            training_context_length=timedelta(days=1),
            predict_sample_interval=timedelta(minutes=15),
        )

    @property
    def quantiles(self) -> list[Quantile]:
        """Return the quantiles this forecaster produces."""
        return self._quantiles

    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """No-op: backtest output already exists."""

    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:  # noqa: ARG002, PLR6301
        """No-op: backtest output already exists.

        Returns:
            Always None.
        """
        return None


def stub_factory(_context: BenchmarkContext, _target: BenchmarkTarget) -> _QuantileStub:
    """Factory that returns a quantile-only stub (backtesting is skipped).

    Returns:
        Stub forecaster with quantile info only.
    """
    return _QuantileStub(quantiles=PREDICTION_QUANTILES)


if __name__ == "__main__":
    storage = LocalBenchmarkStorage(base_path=OUTPUT_PATH / "MyForecasts")

    # ---------------------------------------------------------------
    # Step 1: Save your pre-existing forecasts for each target
    # ---------------------------------------------------------------
    # The pipeline needs one parquet per target, saved via storage.
    # Get the list of targets from the benchmark runner:
    runner = create_custom_benchmark_runner(storage=storage)
    targets = runner.target_provider.get_targets(["solar_park"])

    for target in targets:
        if storage.has_backtest_output(target):
            _logger.info("Predictions already saved for %s, skipping", target.name)
            continue

        # --- Replace this block with your actual forecast loading logic ---
        # This example creates dummy constant predictions as a placeholder.
        idx = pd.date_range(target.benchmark_start, target.benchmark_end, freq="15min")
        dummy_predictions = format_predictions(
            timestamps=idx,
            available_at=pd.Series(idx - timedelta(days=1), index=idx),
            predictions={q: [42.0] * len(idx) for q in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]},
        )
        # --- End of placeholder ---

        storage.save_backtest_output(target=target, output=dummy_predictions)
        _logger.info("Saved predictions for %s", target.name)

    # ---------------------------------------------------------------
    # Step 2: Run the pipeline -- backtesting is auto-skipped
    # ---------------------------------------------------------------
    # The pipeline detects existing backtest files and jumps to evaluation + analysis.
    runner.run(
        forecaster_factory=stub_factory,
        run_name="my_forecasts",
        n_processes=N_PROCESSES,
        filter_args=["solar_park"],
    )
