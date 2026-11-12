# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Comprehensive evaluation pipeline for probabilistic forecasting models.

Segments prediction data across multiple dimensions (availability times, lead times,
time windows) and computes performance metrics for each subset. Enables detailed
analysis of how model performance varies across different operational conditions.
"""

import logging
from collections.abc import Iterator
from datetime import datetime, timedelta
from itertools import chain
from typing import cast

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
from openstef_beam.evaluation.reference_forecast import ReferenceForecastConfig, build_reference_forecast
from openstef_beam.evaluation.window_iterators import iterate_subsets_by_window
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import (
    ForecastDataset,
    ForecastInputDataset,
    TimeSeriesDataset,
    VersionedTimeSeriesDataset,
)
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

    reference_forecast: ReferenceForecastConfig | None = Field(
        default=None,
        description="Reference for skill scores such as CRPSS; unset references yield NaN",
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
        reference_date: datetime | None = None,
    ) -> EvaluationReport:
        """Evaluates predictions against ground truth.

        Segments data by available_at and lead_time configurations, then computes
        metrics for each subset.

        Args:
            predictions: Forecasted values with versioning information.
            ground_truth: Actual observed values for comparison.
            target_column: Name of the target column in ground truth dataset.
            evaluation_mask: Optional datetime index to limit evaluation period.
            reference_date: Reference date for windowed evaluation. When provided, enables
                evaluation focused on a specific time period around this date.

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
        ground_truth_data = ForecastInputDataset.from_timeseries(
            dataset=ground_truth.select_version(),
            target_column=target_column,
        ).pipe_pandas(pd.DataFrame.dropna)
        for filtering, subset, issue_times in self._iterate_subsets(
            predictions=predictions,
            ground_truth_data=ground_truth_data,
            evaluation_mask=evaluation_mask,
            target_column=target_column,
        ):
            if subset.index.empty:
                _logger.warning("No overlapping data for filtering %s. Skipping.", filtering)
                continue

            subset_metrics = self._evaluate_subset(
                subset=subset,
                reference=self._build_reference(
                    subset=subset,
                    measurements=ground_truth_data.target_series,
                    issue_times=issue_times,
                ),
                reference_date=reference_date,
            )

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
        reference_date: datetime | None = None,
    ) -> EvaluationSubsetReport:
        """Evaluates a single evaluation subset.

        Computes metrics for the provided subset without additional filtering.

        Args:
            filtering: The filtering criteria describing this subset.
            predictions: TimeSeriesDataset containing the predicted values.
            evaluation_mask: Optional datetime index to limit evaluation period.
            reference_date: Reference date for windowed evaluation. When provided, enables
                evaluation focused on a specific time period around this date.

        Returns:
            EvaluationSubsetReport containing computed metrics for the subset.
        """
        subset = predictions.filter_quantiles(quantiles=self.quantiles)
        if evaluation_mask is not None:
            subset = subset.filter_index(evaluation_mask)

        subset_metrics = self._evaluate_subset(subset=subset, reference=None, reference_date=reference_date)

        return EvaluationSubsetReport(
            filtering=filtering,
            subset=subset,
            metrics=subset_metrics,
        )

    def _iterate_subsets(
        self,
        predictions: TimeSeriesMixin,
        ground_truth_data: ForecastInputDataset,
        target_column: str,
        evaluation_mask: pd.DatetimeIndex | None = None,
    ) -> Iterator[tuple[Filtering, ForecastDataset, pd.Series | None]]:
        """Yields evaluation subsets filtered by available_at and lead_time.

        Yields:
            Tuples of (filter_criteria, evaluation_subset, issue_times), where issue_times holds the
            time each forecast in the subset was issued, or None if the predictions are not versioned.
        """
        by_available_at = ((at, predictions.filter_by_available_at(at)) for at in self.config.available_ats)
        by_lead_time = ((lt, predictions.filter_by_lead_time(lt)) for lt in self.config.lead_times)
        filtered_predictions: Iterator[tuple[Filtering, TimeSeriesMixin]] = chain(by_available_at, by_lead_time)
        for filtering, versions in filtered_predictions:
            issue_times = self._issue_times(versions)
            predictions_filtered = versions.select_version()
            if evaluation_mask is not None:
                predictions_filtered = predictions_filtered.filter_index(evaluation_mask)

            # Remove target column from predictions to avoid duplication
            if target_column in predictions_filtered.data.columns:
                predictions_filtered = predictions_filtered.pipe_pandas(lambda df: df.drop(columns=[target_column]))

            subset = ForecastDataset(
                data=ground_truth_data.data.join(predictions_filtered.data, how="inner"),
                sample_interval=predictions.sample_interval,
                target_column=target_column,
            ).filter_quantiles(quantiles=self.quantiles)
            yield filtering, subset, None if issue_times is None else issue_times.reindex(subset.index)

    @staticmethod
    def _issue_times(versions: TimeSeriesMixin) -> pd.Series | None:
        """Availability of the version that ``select_version`` keeps for each timestamp.

        Returns:
            Issue time per timestamp, or None if no part of the dataset is versioned.
        """
        parts = versions.data_parts if isinstance(versions, VersionedTimeSeriesDataset) else [versions]
        issue_times = [
            available_at[~part.index.duplicated(keep="first")]
            for part in parts
            if isinstance(part, TimeSeriesDataset) and (available_at := part.available_at_series) is not None
        ]
        return pd.concat(issue_times, axis=1).max(axis=1) if issue_times else None

    def _build_reference(
        self,
        subset: ForecastDataset,
        measurements: pd.Series,
        issue_times: pd.Series | None,
    ) -> ForecastDataset | None:
        if self.config.reference_forecast is None:
            return None
        if issue_times is None:
            _logger.warning("Predictions are not versioned; skipping the reference forecast.")
            return None

        return build_reference_forecast(
            config=self.config.reference_forecast,
            measurements=measurements,
            subset=subset,
            issue_times=issue_times,
        )

    def _evaluate_subset(
        self,
        subset: ForecastDataset,
        reference: ForecastDataset | None,
        reference_date: datetime | None = None,
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
                windowed_metrics.extend(
                    [
                        SubsetMetric(
                            window=window,
                            timestamp=window_timestamp,
                            metrics=merge_quantile_metrics(
                                [
                                    provider(
                                        window_subset,
                                        reference.filter_index(window_subset.index) if reference is not None else None,
                                    )
                                    for provider in self.window_metric_providers
                                ]
                            ),
                        )
                        for window_timestamp, window_subset in iterate_subsets_by_window(
                            subset=subset,
                            window=window,
                            reference_date=reference_date,
                        )
                    ]
                )

        if reference_date is None:
            # When reference data is set, user likely wants metrics for that specific time frame
            # So we skip global metrics in that case.
            windowed_metrics.append(
                SubsetMetric(
                    window="global",
                    timestamp=cast("pd.Series[pd.Timestamp]", subset.index).min().to_pydatetime(),
                    metrics=merge_quantile_metrics(
                        [provider(subset, reference) for provider in self.global_metric_providers]
                    ),
                )
            )
        return windowed_metrics


__all__ = [
    "EvaluationConfig",
    "EvaluationPipeline",
]
