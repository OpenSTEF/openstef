# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Comprehensive evaluation pipeline for probabilistic forecasting models.

Segments prediction data across multiple dimensions (availability times, lead times,
time windows) and computes performance metrics for each subset. Enables detailed
analysis of how model performance varies across different operational conditions.
"""

import logging
from collections.abc import Iterator
from datetime import timedelta

import pandas as pd
from pydantic import Field

from openstef_beam.evaluation.metric_providers import MetricProvider, ObservedProbabilityProvider
from openstef_beam.evaluation.models import (
    EvaluationReport,
    EvaluationSubsetReport,
    Filtering,
    SubsetMetric,
    Window,
)
from openstef_beam.evaluation.models.subset import merge_quantile_metrics
from openstef_beam.evaluation.window_iterators import iterate_subsets_by_window
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.datasets.mixins import TimeSeriesMixin
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import AvailableAt, LeadTime, Quantile

_logger = logging.getLogger(__name__)


class EvaluationConfig(BaseConfig):
    """Configuration for the evaluation pipeline.

    Controls how predictions are filtered, grouped, and analyzed across different time dimensions.
    """

    available_ats: list[AvailableAt] = Field(
        default=[AvailableAt.from_string("D-1T06:00")],
        description="Time points when predictions become available relative to the target date",
    )
    lead_times: list[LeadTime] = Field(
        default=[LeadTime.from_string("PT36H")],
        description="Time intervals between prediction generation and the target timestamp",
    )

    windows: list[Window] = Field(
        default=[Window(lag=timedelta(hours=0), size=timedelta(days=21))],
        description="Time windows for rolling evaluation periods",
    )


class EvaluationPipeline:
    """Pipeline for evaluating probabilistic forecasting models.

    Computes metrics across various dimensions:
    - Prediction availability times
    - Lead times
    - Time windows
    - Global and windowed metrics

    Always includes observed probability as a calibration metric.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        quantiles: list[Quantile],
        window_metric_providers: list[MetricProvider],
        global_metric_providers: list[MetricProvider],
    ) -> None:
        """Initializes the pipeline with configuration and metric providers.

        Automatically adds ObservedProbabilityProvider to global metrics to ensure
        calibration is always evaluated.

        Args:
            config: Configuration for evaluation pipeline with time dimensions.
            quantiles: List of quantiles to evaluate, must include 0.5 (median).
            window_metric_providers: Metrics to compute for time windows.
            global_metric_providers: Metrics to compute for entire dataset.

        Raises:
            ValueError: If quantiles list does not include 0.5 (median quantile).
        """
        if 0.5 not in quantiles:  # noqa: PLR2004 0.5 is the median quantile
            raise ValueError("Quantiles must include 0.5 for median evaluation.")

        super().__init__()
        self.config = config
        self.quantiles = quantiles
        self.window_metric_providers = window_metric_providers
        self.global_metric_providers = [
            *global_metric_providers,
            ObservedProbabilityProvider(),
        ]

    def run(
        self,
        predictions: TimeSeriesMixin,
        ground_truth: TimeSeriesMixin,
        target_column: str,
        evaluation_mask: pd.DatetimeIndex | None = None,
    ) -> EvaluationReport:
        """Evaluates predictions against ground truth.

        Segments data by available_at and lead_time configurations, then computes
        metrics for each subset.

        Args:
            predictions: Forecasted values with versioning information.
            ground_truth: Actual observed values for comparison.
            target_column: Name of the target column in ground truth dataset.
            evaluation_mask: Optional datetime index to limit evaluation period.

        Returns:
            EvaluationReport containing metrics for each subset, organized by
            filtering criteria such as lead time windows and availability timestamps.

        Raises:
            ValueError: If predictions and ground truth have different sample intervals.
            MissingColumnsError: If any configured quantile columns are missing from predictions.
        """
        # Validate that the predictions and ground truth datasets have the same sample interval
        if predictions.sample_interval != ground_truth.sample_interval:
            raise ValueError("Predictions and ground truth must have the same sample interval.")

        # Validate that the required quantiles are present in the predictions
        quantile_columns = [quantile.format() for quantile in self.quantiles]
        missing_quantiles = set(quantile_columns) - set(predictions.feature_names)
        if missing_quantiles:
            raise MissingColumnsError(missing_columns=list(missing_quantiles))

        subsets: list[EvaluationSubsetReport] = []
        for filtering, subset in self._iterate_subsets(
            predictions=predictions,
            ground_truth=ground_truth,
            evaluation_mask=evaluation_mask,
            target_column=target_column,
        ):
            if subset.index.empty:
                _logger.warning("No overlapping data for filtering %s. Skipping.", filtering)
                continue

            subset_metrics = self._evaluate_subset(subset=subset)

            subsets.append(
                EvaluationSubsetReport(
                    filtering=filtering,
                    subset=subset,
                    metrics=subset_metrics,
                )
            )

        return EvaluationReport(
            subset_reports=subsets,
        )

    def run_for_subset(
        self,
        filtering: Filtering,
        predictions: ForecastDataset,
        evaluation_mask: pd.DatetimeIndex | None = None,
    ) -> EvaluationSubsetReport:
        """Evaluates a single evaluation subset.

        Computes metrics for the provided subset without additional filtering.

        Args:
            filtering: The filtering criteria describing this subset.
            predictions: TimeSeriesDataset containing the predicted values.
            evaluation_mask: Optional datetime index to limit evaluation period.

        Returns:
            EvaluationSubsetReport containing computed metrics for the subset.
        """
        subset = predictions.filter_quantiles(quantiles=self.quantiles)
        if evaluation_mask is not None:
            subset = subset.filter_index(evaluation_mask)

        subset_metrics = self._evaluate_subset(subset=subset)

        return EvaluationSubsetReport(
            filtering=filtering,
            subset=subset,
            metrics=subset_metrics,
        )

    def _iterate_subsets(
        self,
        predictions: TimeSeriesMixin,
        ground_truth: TimeSeriesMixin,
        target_column: str,
        evaluation_mask: pd.DatetimeIndex | None = None,
    ) -> Iterator[tuple[Filtering, ForecastDataset]]:
        """Yields evaluation subsets filtered by available_at and lead_time.

        For ground truth, the data with the latest lead time is used.

        Yields:
            Tuples of (filter_criteria, evaluation_subset)
        """
        ground_truth_data = ForecastInputDataset.from_timeseries(
            dataset=ground_truth.select_version(),
            target_column=target_column,
        )

        for available_at in self.config.available_ats:
            predictions_filtered = predictions.filter_by_available_at(available_at=available_at).select_version()
            if evaluation_mask is not None:
                predictions_filtered = predictions_filtered.filter_index(evaluation_mask)

            yield (
                available_at,
                ForecastDataset(
                    data=ground_truth_data.data.join(predictions_filtered.data, how="inner"),
                    sample_interval=predictions.sample_interval,
                    target_column=target_column,
                ).filter_quantiles(quantiles=self.quantiles),
            )

        for lead_time in self.config.lead_times:
            predictions_filtered = predictions.filter_by_lead_time(lead_time=lead_time).select_version()
            if evaluation_mask is not None:
                predictions_filtered = predictions_filtered.filter_index(evaluation_mask)

            yield (
                lead_time,
                ForecastDataset(
                    data=ground_truth_data.data.join(predictions_filtered.data, how="inner"),
                    sample_interval=predictions.sample_interval,
                    target_column=target_column,
                ).filter_quantiles(quantiles=self.quantiles),
            )

    def _evaluate_subset(
        self,
        subset: ForecastDataset,
    ) -> list[SubsetMetric]:
        """Computes metrics for a given evaluation subset.

        Applies window-specific metrics to each time window and adds a global
        evaluation across the entire subset. Each window generates one SubsetMetric
        per timestamp, plus one global metric.

        Returns:
            List of SubsetMetric objects containing computed metrics for each window
            and timestamp combination, plus the global metrics.
        """
        windowed_metrics: list[SubsetMetric] = []

        if len(self.window_metric_providers) > 0:
            for window in self.config.windows:
                windowed_metrics.extend([
                    SubsetMetric(
                        window=window,
                        timestamp=window_timestamp,
                        metrics=merge_quantile_metrics([
                            provider(window_subset) for provider in self.window_metric_providers
                        ]),
                    )
                    for window_timestamp, window_subset in iterate_subsets_by_window(
                        subset=subset,
                        window=window,
                    )
                ])

        windowed_metrics.append(
            SubsetMetric(
                window="global",
                timestamp=subset.index.min().to_pydatetime(),  # type: ignore[reportUnknownMemberType]
                metrics=merge_quantile_metrics([provider(subset) for provider in self.global_metric_providers]),
            )
        )
        return windowed_metrics


__all__ = [
    "EvaluationConfig",
    "EvaluationPipeline",
]
