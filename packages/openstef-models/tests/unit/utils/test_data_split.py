# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the data splitting utilities."""

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.utils import data_split


def _with_normalize(dataset: TimeSeriesDataset) -> TimeSeriesDataset:
    """Attach a normalize helper so split helpers operate on the dataset index."""

    def _normalize_index() -> pd.DatetimeIndex:
        return pd.DatetimeIndex(dataset.index).normalize()

    dataset.normalize = _normalize_index  # type: ignore[attr-defined]
    return dataset


@pytest.fixture
def simple_daily_dataset() -> TimeSeriesDataset:
    """Simple daily dataset without versioning metadata."""
    index = pd.date_range("2025-01-01", periods=10, freq="1D")
    data = pd.DataFrame(
        {
            "load": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
            "temperature": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        },
        index=index,
    )
    dataset = TimeSeriesDataset(data=data, sample_interval=timedelta(days=1))
    return _with_normalize(dataset)


@pytest.fixture
def versioned_daily_dataset() -> TimeSeriesDataset:
    """Daily dataset with availability column to exercise stratified splitting."""
    index = pd.date_range("2025-02-01", periods=12, freq="1D")
    load = [50, 60, 70, 80, 95, 100, 105, 110, 160, 170, 180, 190]
    data = pd.DataFrame(
        {
            "load": load,
            "available_at": index + pd.Timedelta(minutes=30),
        },
        index=index,
    )
    dataset = TimeSeriesDataset(data=data, sample_interval=timedelta(days=1))
    dataset.data["horizon"] = pd.Timedelta(hours=1)
    return _with_normalize(dataset)


def test_split_by_date__disjoint_partitions(simple_daily_dataset: TimeSeriesDataset) -> None:
    """split_by_date should produce contiguous train/test segments."""
    # Arrange
    split_point = pd.Timestamp("2025-01-06")

    # Act
    train_dataset, test_dataset = data_split.split_by_date(simple_daily_dataset, split_point)

    expected_train_index = [timestamp for timestamp in simple_daily_dataset.index if timestamp < split_point]
    expected_test_index = [timestamp for timestamp in simple_daily_dataset.index if timestamp >= split_point]

    # Assert - training split contains only timestamps before the split point
    assert list(train_dataset.index) == expected_train_index
    # Assert - test split contains the remaining timestamps
    assert list(test_dataset.index) == expected_test_index
    # Assert - no samples lost during the split
    assert len(train_dataset.data) + len(test_dataset.data) == len(simple_daily_dataset.data)


def test_split_by_dates__selects_requested_days(
    simple_daily_dataset: TimeSeriesDataset,
) -> None:
    """split_by_dates should isolate requested days and keep the rest for training."""
    # Arrange
    test_dates = pd.DatetimeIndex(["2025-01-02", "2025-01-05", "2025-01-09"])

    # Act
    train_dataset, test_dataset = data_split.split_by_dates(simple_daily_dataset, test_dates)

    # Assert - requested days move to the test split
    assert list(test_dataset.index) == list(test_dates)
    # Assert - training split retains the complementary set of days
    train_expected = [timestamp for timestamp in simple_daily_dataset.index if timestamp not in set(test_dates)]
    assert list(train_dataset.index) == train_expected


def test_chronological_train_test_split__respects_fraction(
    simple_daily_dataset: TimeSeriesDataset,
) -> None:
    """chronological_train_test_split should honour the test fraction using latest samples."""
    # Arrange
    test_fraction = 0.3

    # Act
    train_dataset, test_dataset = data_split.chronological_train_test_split(simple_daily_dataset, test_fraction)

    # Assert - test contains the most recent dates according to the fraction
    assert list(test_dataset.index) == list(simple_daily_dataset.index[-3:])
    # Assert - train contains the complementary leading segment
    assert list(train_dataset.index) == list(simple_daily_dataset.index[:-3])


def test_stratified_train_test_split__includes_extreme_days(
    versioned_daily_dataset: TimeSeriesDataset,
) -> None:
    """stratified_train_test_split should always include extreme min/max days in the test set."""
    # Arrange
    test_fraction = 0.25

    # Act
    train_dataset, test_dataset = data_split.stratified_train_test_split(
        dataset=versioned_daily_dataset,
        test_fraction=test_fraction,
        stratification_fraction=0.2,
        target_column="load",
        random_state=7,
        min_days_for_stratification=4,
    )

    # Assert - high extreme days appear in the test split
    target_series = versioned_daily_dataset.select_features(["load"]).select_version().data["load"]
    extreme_max_days = pd.DatetimeIndex(target_series.nlargest(3).index).normalize()
    normalized_test_index = pd.DatetimeIndex(test_dataset.index).normalize()
    assert not set(normalized_test_index).isdisjoint(set(extreme_max_days))
    # Assert - low extreme days appear in the test split
    extreme_min_days = pd.DatetimeIndex(target_series.nsmallest(3).index).normalize()
    assert not set(normalized_test_index).isdisjoint(set(extreme_min_days))
    # Assert - combined data still matches the original dataset
    combined = pd.concat([train_dataset.data, test_dataset.data]).sort_index()
    assert combined.equals(versioned_daily_dataset.data)


def test_train_val_test_split__partitions_dataset(
    simple_daily_dataset: TimeSeriesDataset,
) -> None:
    """train_val_test_split should return contiguous and exhaustive partitions."""
    # Arrange
    val_fraction = 0.1
    test_fraction = 0.2

    # Act
    train_dataset, val_dataset, test_dataset = data_split.train_val_test_split(
        dataset=simple_daily_dataset,
        split_func=data_split.chronological_train_test_split,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    # Assert - partition sizes align with requested fractions
    assert len(train_dataset.data) == 7
    assert len(val_dataset.data) == 1
    assert len(test_dataset.data) == 2
    # Assert - partitions cover the entire time range without overlap
    combined_index = train_dataset.index.append(val_dataset.index).append(test_dataset.index)
    assert combined_index.to_series().equals(simple_daily_dataset.index.to_series())
