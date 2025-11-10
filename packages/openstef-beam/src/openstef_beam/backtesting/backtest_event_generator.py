# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Event scheduling engine for realistic backtesting simulations.

Determines when training and prediction events should occur during backtesting,
respecting data availability constraints and operational scheduling requirements.
Acts as the temporal coordinator between data availability and model operations.
"""

from collections.abc import Iterator
from datetime import datetime, time, timedelta

import pandas as pd
from pydantic import Field

from openstef_beam.backtesting.backtest_event import BacktestEvent, BacktestEventBatch
from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig
from openstef_core.base_model import BaseModel
from openstef_core.utils.datetime import align_datetime_to_time
from openstef_core.utils.itertools import merge_iterators


class BacktestEventGenerator(BaseModel):
    """Component for generating ordered sequences of backtest events.

    Generates train and predict events based on configured intervals, ensuring that
    each event has sufficient context data. Events are ordered chronologically with
    train events preceding predict events at the same timestamp.
    """

    start: datetime
    end: datetime

    index: pd.DatetimeIndex = Field(..., description="Timestamps for the prediction target series.")
    sample_interval: timedelta = Field(..., description="Interval between timestamps.")

    predict_interval: timedelta = Field(..., description="Interval between predictions.")
    train_interval: timedelta = Field(..., description="Interval between retrains.")
    align_time: time = Field(
        ..., description="Time alignment for predictions, ensuring they are aligned to this interval."
    )
    forecaster_config: BacktestForecasterConfig = Field(..., description="Model interface configuration.")

    def iterate(self) -> Iterator[BacktestEvent]:
        """Creates an ordered iterator of train and predict events.

        Combines training and prediction events in chronological order, with
        training events preceding prediction events at the same timestamp.
        If the model doesn't require training, only prediction events are returned.

        Returns:
           An iterator of chronologically ordered BacktestEvents.
        """
        if not self.forecaster_config.requires_training:
            return self._predict_iterator()

        train_events = list(self._train_iterator())
        if len(train_events) == 0:
            return iter([])

        trained_timestamp = train_events[0].timestamp
        predict_events = [event for event in self._predict_iterator() if event.timestamp >= trained_timestamp]

        return merge_iterators(
            it1=iter(train_events),
            it2=iter(predict_events),
            compare=self._cmp_events,
        )

    @staticmethod
    def _cmp_events(a: BacktestEvent, b: BacktestEvent) -> int:
        """Compares two BacktestEvents for ordering.

        First orders by timestamp, then by event type (train before predict).

        Returns:
            -1 if a should come before b, 1 if a should come after b, 0 if equal.
        """
        if a.timestamp < b.timestamp:
            return -1
        if a.timestamp > b.timestamp:
            return 1

        order = {"train": 0, "predict": 1}
        return order[a.type] - order[b.type]

    def _predict_iterator(self) -> Iterator[BacktestEvent]:
        """Generates prediction events at configured intervals.

        Creates predict events that have sufficient forecast context coverage
        and occur at regular intervals as specified by predict_interval.

        Yields:
           BacktestEvent: Prediction events with sufficient context coverage.
        """
        end_time = self.end
        current_time = align_datetime_to_time(self.start, self.align_time, mode="ceil")

        while current_time <= end_time:
            horizon_end = current_time + self.forecaster_config.horizon_min_length
            if horizon_end > end_time:
                break

            forecast_context_start = current_time - self.forecaster_config.predict_context_length
            forecast_context_end = current_time
            forecast_context_coverage = self._calculate_coverage(forecast_context_start, forecast_context_end)
            if forecast_context_coverage >= self.forecaster_config.predict_context_min_coverage:
                yield BacktestEvent(type="predict", timestamp=current_time)

            current_time += self.predict_interval

    def _train_iterator(self) -> Iterator[BacktestEvent]:
        """Generates training events at configured intervals.

        Creates train events that have sufficient training context coverage
        and occur at regular intervals as specified by train_interval.

        Yields:
           BacktestEvent: Training events with sufficient context coverage.
        """
        end_time = self.end
        current_time = align_datetime_to_time(self.start, self.align_time, mode="ceil")

        while current_time <= end_time:
            horizon_end = current_time + self.forecaster_config.horizon_min_length
            if horizon_end > end_time:
                break

            training_start = max(
                current_time - self.forecaster_config.training_context_length, min(self.index).to_pydatetime()
            )
            training_end = current_time
            training_coverage = self._calculate_coverage(training_start, training_end)
            if training_coverage >= self.forecaster_config.training_context_min_coverage:
                yield BacktestEvent(type="train", timestamp=current_time)

            current_time += self.train_interval

    def _calculate_coverage(self, start: datetime, end: datetime) -> float:
        """Calculates the data coverage ratio within a time window.

        Determines what fraction of expected data points are actually available
        in the provided index within the specified time window. If start >= end,
        coverage is defined as 0.0.

        Args:
            start: The start of the time window.
            end: The end of the time window.

        Returns:
            The ratio of available data points to expected data points.
        """
        if start >= end:
            return 0.0

        num_window_samples = (end - start) / self.sample_interval
        coverage = self.index[(self.index >= pd.Timestamp(start)) & (self.index < pd.Timestamp(end))]
        return len(coverage) / num_window_samples

    @staticmethod
    def iterate_batched(events: list[BacktestEvent], batch_size: int | None = None) -> Iterator[BacktestEventBatch]:
        """Creates an iterator of batched backtest events for efficient processing.

        Groups prediction events into batches up to batch_size, while keeping
        training events as individual batches. This provides a clean interface
        for batch-aware processing without mixing concerns.

        Args:
            events: List of BacktestEvent objects to be processed in batches.
            batch_size: Maximum number of prediction events per batch.
                       If None, all events are returned as individual batches.

        Yields:
            BacktestEventBatch: Batched events ready for processing.
        """
        if batch_size is None or batch_size <= 1:
            # Return individual events as single-element batches
            for event in events:
                yield BacktestEventBatch(events=[event])
            return

        i = 0
        while i < len(events):
            event = events[i]

            if event.type == "train":
                # Training events are always processed individually
                yield BacktestEventBatch(events=[event])
                i += 1
            else:
                # Collect consecutive prediction events up to batch_size
                batch_events: list[BacktestEvent] = []
                batch_end = min(i + batch_size, len(events))

                for j in range(i, batch_end):
                    if events[j].type != "predict":
                        break
                    batch_events.append(events[j])

                if batch_events:
                    yield BacktestEventBatch(events=batch_events)
                    i += len(batch_events)
                else:
                    i += 1
