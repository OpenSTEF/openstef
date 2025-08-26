# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Backtesting pipeline for evaluating energy forecasting models.

Simulates realistic forecasting scenarios by replaying historical data with
proper temporal constraints. Executes prediction and retraining schedules
that mirror operational deployment conditions, ensuring evaluation results
accurately reflect real-world model performance.
"""

import logging
from datetime import datetime, time, timedelta
from typing import cast

import pandas as pd
from pydantic import Field
from tqdm.auto import tqdm

from openstef_beam.backtesting.backtest_event import BacktestEvent
from openstef_beam.backtesting.backtest_event_generator import BacktestEventGenerator
from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestBatchForecasterMixin, BacktestForecasterMixin
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.datasets.versioned_timeseries import concat_featurewise, restrict_horizon

_logger = logging.getLogger(__name__)


class BacktestConfig(BaseConfig):
    """Configuration for backtesting energy forecasting models."""

    prediction_sample_interval: timedelta = Field(
        default=timedelta(minutes=15),
        description="Time interval between prediction samples in the output forecast",
    )
    predict_interval: timedelta = Field(
        default=timedelta(hours=6),
        description="Time interval between generating new predictions during backtesting",
    )
    train_interval: timedelta = Field(
        default=timedelta(days=7),
        description="Time interval between model retraining events",
    )
    align_time: time = Field(
        default=time.fromisoformat("00:00+00"),
        description="Reference time for aligning prediction schedules to regular intervals",
    )


class BacktestPipeline:
    """Pipeline for conducting realistic backtesting of energy forecasting models.

    This class orchestrates the backtesting process by simulating the operational
    environment where forecasts are generated at regular intervals with limited
    historical data availability. It supports both single and batch prediction modes
    and handles periodic model retraining.

    Attributes:
        config: Configuration parameters for the backtesting process.
        forecaster: The forecasting model implementing either BacktestForecasterMixin
            or BacktestBatchForecasterMixin interface.
        start: Start datetime for the backtesting period.
        end: End datetime for the backtesting period.

    Note:
        The pipeline ensures temporal consistency by preventing data leakage and
        respecting the operational constraints that would exist in a real-time
        forecasting system.
    """

    config: BacktestConfig
    forecaster: BacktestForecasterMixin
    start: datetime
    end: datetime

    def __init__(
        self,
        config: BacktestConfig,
        forecaster: BacktestForecasterMixin,
    ) -> None:
        """Initialize the backtesting pipeline.

        Args:
            config: Backtesting configuration including prediction and training intervals.
            forecaster: Model implementing the required forecasting interface.

        Raises:
            ValueError: If the prediction sample intervals don't match between
                config and forecaster.
        """
        if config.prediction_sample_interval != forecaster.config.predict_sample_interval:
            raise ValueError(
                "The prediction sample interval of the backtest config must match the model interface's predict sample "
                "interval.",
                config.prediction_sample_interval,
                forecaster.config.predict_sample_interval,
            )

        self.config = config
        self.forecaster = forecaster

    def run(
        self,
        ground_truth: VersionedTimeSeriesMixin,
        predictors: VersionedTimeSeriesMixin,
        start: datetime | None,
        end: datetime | None,
        *,
        show_progress: bool = True,
    ) -> VersionedTimeSeriesDataset:
        """Execute the backtesting simulation and return predictions.

        Runs the complete backtesting process by generating events, processing
        training and prediction operations in chronological order, and collecting
        all predictions into a single versioned dataset.

        Args:
            ground_truth: Historical target values with timestamps.
            predictors: Feature data for model predictions.
            start: Start datetime for backtesting. Uses data minimum if None.
            end: End datetime for backtesting. Uses data maximum if None.
            show_progress: Whether to display progress bar during execution.

        Returns:
            VersionedTimeSeriesDataset containing all predictions with timestamps
            and availability information. Empty dataset if no predictions made.
        """
        min_start = ground_truth.index.min().to_pydatetime()  # type: ignore[reportUnknownMemberType]
        max_end = ground_truth.index.max().to_pydatetime()  # type: ignore[reportUnknownMemberType]

        # Prepare the input data
        dataset = concat_featurewise(
            datasets=[ground_truth, predictors],
            mode="outer",
        )
        _logger.info("Initialized backtest dataset with %d features", len(dataset.feature_names))

        # Prepare backtesting events with batch awareness
        event_factory = BacktestEventGenerator(
            start=max(start, min_start) if start else min_start,
            end=min(end, max_end) if end else max_end,
            index=ground_truth.index,
            sample_interval=ground_truth.sample_interval,
            predict_interval=self.config.predict_interval,
            train_interval=self.config.train_interval,
            align_time=self.config.align_time,
            forecaster_config=self.forecaster.config,
        )

        # Determine batch processing approach
        if isinstance(self.forecaster, BacktestBatchForecasterMixin):
            batch_size = self.forecaster.batch_size
            supports_batching = batch_size is not None and batch_size > 1
        else:
            batch_size = None
            supports_batching = False

        _logger.info("Starting the backtest pipeline")
        prediction_list: list[VersionedTimeSeriesDataset] = []

        prediction_list.extend(
            self._process_events(
                event_factory=event_factory,
                dataset=dataset,
                batch_size=batch_size if supports_batching else None,
                show_progress=show_progress,
            )
        )

        _logger.info("Finished backtest pipeline")
        if not prediction_list:
            return VersionedTimeSeriesDataset(
                data=pd.DataFrame({
                    "timestamp": pd.Series(dtype="datetime64[ns]"),
                    "available_at": pd.Series(dtype="datetime64[ns]"),
                }),
                sample_interval=self.config.prediction_sample_interval,
            )

        return VersionedTimeSeriesDataset(
            data=pd.concat([pred.data for pred in prediction_list], axis=0),
            sample_interval=self.config.prediction_sample_interval,
        )

    def _process_train_event(self, event: BacktestEvent, dataset: VersionedTimeSeriesMixin) -> None:
        """Process a single training event."""
        horizon_dataset = restrict_horizon(dataset=dataset, horizon=event.timestamp)
        self.forecaster.fit(horizon_dataset)
        _logger.debug("Processed train event", extra={"event": event})

    def _process_single_prediction(
        self, event: BacktestEvent, dataset: VersionedTimeSeriesMixin
    ) -> list[VersionedTimeSeriesDataset]:
        """Process a single prediction event.

        Args:
            event: Prediction event to process.
            dataset: Time series data for prediction.

        Returns:
            List containing single prediction dataset if successful, empty list otherwise.
        """
        horizon_dataset = restrict_horizon(dataset=dataset, horizon=event.timestamp)
        prediction = self.forecaster.predict_versioned(horizon_dataset)

        if prediction is not None:
            _logger.debug("Processed single prediction")
            return [prediction]
        else:  # noqa: RET505 - kept for readability
            _logger.warning("Model interface returned None for prediction", extra={"event": event})
            return []

    def _process_batch_prediction(
        self, batch_events: list[BacktestEvent], dataset: VersionedTimeSeriesMixin
    ) -> list[VersionedTimeSeriesDataset]:
        """Process a batch of prediction events and return valid predictions.

        Args:
            batch_events: List of prediction events to process as a batch.
            dataset: Time series data for predictions.

        Returns:
            List of valid prediction datasets, excluding any None results.
        """
        if not batch_events:
            return []

        # Process batch
        horizon_datasets = [restrict_horizon(dataset=dataset, horizon=event.timestamp) for event in batch_events]

        batch_predictions = cast(BacktestBatchForecasterMixin, self.forecaster).predict_batch_versioned(
            horizon_datasets
        )
        valid_predictions = [pred for pred in batch_predictions if pred is not None]

        # Log warnings for None predictions
        none_count = len(batch_predictions) - len(valid_predictions)
        if none_count > 0:
            _logger.warning(
                "Model interface returned (%d/%d) None predictions in batch.", none_count, len(batch_events)
            )

        _logger.debug("Processed prediction batch (size=%d)", len(batch_events))
        return valid_predictions

    def _process_events(
        self,
        event_factory: BacktestEventGenerator,
        dataset: VersionedTimeSeriesMixin,
        batch_size: int | None,
        *,
        show_progress: bool = True,
    ) -> list[VersionedTimeSeriesDataset]:
        """Process events using the factory's batching logic.

        Args:
            event_factory: Generator for creating backtest events.
            dataset: Time series data for processing.
            batch_size: Maximum batch size for predictions, None for single processing.
            show_progress: Whether to display progress bar.

        Returns:
            List of all prediction datasets generated during processing.
        """
        predictions: list[VersionedTimeSeriesDataset] = []

        # Get total count for progress bar
        events = list(event_factory.iterate())
        total_events = len(events)
        pbar = tqdm(total=total_events, smoothing=0.0, disable=not show_progress)

        try:
            for event_batch in BacktestEventGenerator.iterate_batched(events=events, batch_size=batch_size):
                if event_batch.is_training:
                    self._process_train_event(event_batch.events[0], dataset)
                elif event_batch.is_prediction:
                    # Use batch processing if batch_size is enabled, single processing otherwise
                    if batch_size is not None:
                        batch_predictions = self._process_batch_prediction(
                            batch_events=event_batch.events, dataset=dataset
                        )
                    else:
                        batch_predictions = self._process_single_prediction(
                            event=event_batch.events[0], dataset=dataset
                        )
                    predictions.extend(batch_predictions)

                pbar.update(len(event_batch))
        finally:
            pbar.close()

        return predictions


__all__ = [
    "BacktestConfig",
    "BacktestPipeline",
]
