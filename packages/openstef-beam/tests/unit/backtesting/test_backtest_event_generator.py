# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, time, timedelta
from typing import Literal

import pandas as pd
import pytest

from openstef_beam.backtesting.backtest_event import BacktestEvent
from openstef_beam.backtesting.backtest_event_generator import BacktestEventGenerator
from openstef_beam.backtesting.backtest_forecaster.mixins import BacktestForecasterConfig


@pytest.fixture
def config() -> BacktestForecasterConfig:
    return BacktestForecasterConfig(
        requires_training=True,
        predict_length=timedelta(hours=24),
        predict_min_length=timedelta(hours=6),
        predict_context_length=timedelta(hours=12),
        predict_context_min_coverage=0.5,
        training_context_length=timedelta(hours=24),
        training_context_min_coverage=0.5,
    )


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", "2025-01-03", freq="1h")


@pytest.fixture
def factory(config: BacktestForecasterConfig, hourly_index: pd.DatetimeIndex) -> BacktestEventGenerator:
    return BacktestEventGenerator(
        start=datetime.fromisoformat("2025-01-01T12:00:00"),
        end=datetime.fromisoformat("2025-01-02T12:00:00"),
        index=hourly_index,
        sample_interval=timedelta(hours=1),
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(hours=12),
        align_time=time.fromisoformat("00:00+00"),
        forecaster_config=config,
    )


@pytest.mark.parametrize(
    ("event1_type", "event1_time", "event2_type", "event2_time", "expected"),
    [
        ("predict", "2025-01-01T10:00:00", "train", "2025-01-01T12:00:00", -1),
        ("train", "2025-01-01T12:00:00", "predict", "2025-01-01T12:00:00", -1),
        ("predict", "2025-01-01T12:00:00", "predict", "2025-01-01T12:00:00", 0),
    ],
)
def test_cmp_events(
    event1_type: Literal["predict", "train"],
    event1_time: str,
    event2_type: Literal["predict", "train"],
    event2_time: str,
    expected: int,
):
    # Arrange
    event1 = BacktestEvent(type=event1_type, timestamp=datetime.fromisoformat(event1_time))
    event2 = BacktestEvent(type=event2_type, timestamp=datetime.fromisoformat(event2_time))

    # Act
    result = BacktestEventGenerator._cmp_events(event1, event2)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("timestamps", "expected_coverage"),
    [
        (["2025-01-01T12:00:00", "2025-01-01T13:00:00", "2025-01-01T14:00:00"], 1.0),
        (["2025-01-01T12:00:00", "2025-01-01T14:00:00"], 2 / 3),  # Missing 13:00
        ([], 0.0),
    ],
)
def test_calculate_coverage(config: BacktestForecasterConfig, timestamps: list[str], expected_coverage: float):
    # Arrange
    index = pd.DatetimeIndex(timestamps)
    factory = BacktestEventGenerator(
        start=datetime.fromisoformat("2025-01-01T12:00:00"),
        end=datetime.fromisoformat("2025-01-01T15:00:00"),
        index=index,
        sample_interval=timedelta(hours=1),
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(hours=12),
        align_time=time.fromisoformat("00:00+00"),
        forecaster_config=config,
    )

    # Act
    coverage = factory._calculate_coverage(
        datetime.fromisoformat("2025-01-01T12:00:00"), datetime.fromisoformat("2025-01-01T15:00:00")
    )

    # Assert
    assert coverage == expected_coverage


@pytest.mark.parametrize(
    ("method", "event_type"),
    [
        ("_predict_iterator", "predict"),
        ("_train_iterator", "train"),
    ],
)
def test_iterators_generate_correct_event_types(factory: BacktestEventGenerator, method: str, event_type: str):
    # Arrange & Act
    events = list(getattr(factory, method)())

    # Assert
    assert len(events) > 0
    assert all(event.type == event_type for event in events)


def test_iterate_without_training_only_predicts(hourly_index: pd.DatetimeIndex):
    # Arrange
    config = BacktestForecasterConfig(
        requires_training=False,
        predict_length=timedelta(hours=24),
        predict_min_length=timedelta(hours=6),
        predict_context_length=timedelta(hours=12),
        predict_context_min_coverage=0.8,
        training_context_length=timedelta(hours=24),
        training_context_min_coverage=0.8,
    )
    factory = BacktestEventGenerator(
        start=datetime.fromisoformat("2025-01-01T12:00:00"),
        end=datetime.fromisoformat("2025-01-02T12:00:00"),
        index=hourly_index,
        sample_interval=timedelta(hours=1),
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(hours=12),
        align_time=time.fromisoformat("00:00+00"),
        forecaster_config=config,
    )

    # Act
    events = list(factory.iterate())

    # Assert
    assert len(events) > 0
    assert all(event.type == "predict" for event in events)


def test_iterate_with_training_starts_with_train_event(factory: BacktestEventGenerator):
    # Arrange & Act
    events = list(factory.iterate())

    # Assert
    assert len(events) > 0
    assert events[0].type == "train"

    # Both event types should be present
    event_types = {event.type for event in events}
    assert event_types == {"train", "predict"}


def test_iterate_returns_empty_when_insufficient_time():
    # Arrange
    config = BacktestForecasterConfig(
        requires_training=True,
        predict_length=timedelta(hours=24),
        predict_min_length=timedelta(hours=6),
        predict_context_length=timedelta(hours=1),
        predict_context_min_coverage=0.8,
        training_context_length=timedelta(days=10),  # Impossibly long
        training_context_min_coverage=0.8,
    )
    factory = BacktestEventGenerator(
        start=datetime.fromisoformat("2025-01-01T12:00:00"),
        end=datetime.fromisoformat("2025-01-01T18:00:00"),  # Too short
        index=pd.date_range("2025-01-01", "2025-01-03", freq="1h"),
        sample_interval=timedelta(hours=1),
        predict_interval=timedelta(hours=6),
        train_interval=timedelta(hours=12),
        align_time=time.fromisoformat("00:00+00"),
        forecaster_config=config,
    )

    # Act
    events = list(factory.iterate())

    # Assert
    assert len(events) == 0


def test_insufficient_coverage_filters_out_events():
    # Arrange
    sparse_index = pd.DatetimeIndex(["2025-01-01T12:00:00", "2025-01-01T18:00:00"])
    config = BacktestForecasterConfig(
        requires_training=False,
        predict_length=timedelta(hours=24),
        predict_min_length=timedelta(hours=1),
        predict_context_length=timedelta(hours=6),
        predict_context_min_coverage=0.9,  # High requirement
        training_context_length=timedelta(hours=24),
        training_context_min_coverage=0.9,
    )
    factory = BacktestEventGenerator(
        start=datetime.fromisoformat("2025-01-01T18:00:00"),
        end=datetime.fromisoformat("2025-01-01T20:00:00"),
        index=sparse_index,
        sample_interval=timedelta(hours=1),
        predict_interval=timedelta(hours=1),
        train_interval=timedelta(hours=12),
        align_time=time.fromisoformat("00:00+00"),
        forecaster_config=config,
    )

    # Act
    events = list(factory.iterate())

    # Assert
    assert len(events) == 0
