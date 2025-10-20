# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for data splitting functionality."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pydantic
import pytest
from openstef_core.datasets.utils.data_split import (
    ChronologicalTrainTestSplitter,
    StratifiedTrainTestSplitter,
    split_by_date,
    split_by_dates,
)

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.exceptions import NotFittedError


@pytest.fixture
def sample_daily_dataset() -> TimeSeriesDataset:
    """Create a TimeSeriesDataset for testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="1D")
    data = pd.DataFrame(
        {
            "load": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            "temperature": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        },
        index=dates,
    )
    return TimeSeriesDataset(data=data, sample_interval=timedelta(days=1))


@pytest.fixture
def sample_versioned_dataset() -> VersionedTimeSeriesDataset:
    """Create a VersionedTimeSeriesDataset for testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="1D")
    data = pd.DataFrame({
        "timestamp": dates,
        "available_at": dates + pd.Timedelta(minutes=5),
        "load": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "temperature": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    })
    return VersionedTimeSeriesDataset.from_dataframe(data, timedelta(days=1))


@pytest.fixture
def stratified_daily_dataset() -> TimeSeriesDataset:
    """Create a TimeSeriesDataset with stratified daily data."""
    dates = pd.date_range("2023-01-01", periods=20, freq="1D")
    # Create deterministic load pattern with clear extreme days:
    # Days 0-3: low values (50-80) - should be selected as min days
    # Days 4-15: medium values (100-115) - should be other days
    # Days 16-19: high values (150-180) - should be selected as max days
    load_values = [
        50,
        60,
        70,
        80,  # 4 min days
        *range(100, 112),  # 12 medium days (100-111)
        150,
        160,
        170,
        180,  # 4 max days
    ]

    data = pd.DataFrame(
        {
            "load": load_values,
            "temperature": [20] * 20,  # Constant temperature for simplicity
        },
        index=dates,
    )
    return TimeSeriesDataset(data=data, sample_interval=timedelta(days=1))


# Tests for split_by_date function
def test_split_by_date__splits_correctly(sample_daily_dataset: TimeSeriesDataset):
    """Test that split_by_date splits data at the specified date."""
    # Arrange
    split_date = pd.Timestamp("2023-01-05")

    # Act
    train_dataset, test_dataset = split_by_date(sample_daily_dataset, split_date)

    # Assert
    # Check lengths
    assert len(train_dataset.data) == 4  # First 4 days
    assert len(test_dataset.data) == 6  # Last 6 days

    # Check that concatenated indexes equal original (verifies no data loss and correct split)
    combined_index = train_dataset.index.append(test_dataset.index)
    assert combined_index.to_series().equals(sample_daily_dataset.index.to_series())


def test_split_by_date__versioned_dataset(sample_versioned_dataset: VersionedTimeSeriesDataset):
    """Test split_by_date with VersionedTimeSeriesDataset."""
    # Arrange
    split_date = pd.Timestamp("2023-01-05")

    # Act
    train_dataset, test_dataset = split_by_date(sample_versioned_dataset, split_date)

    # Assert
    assert sum(len(part.data) for part in train_dataset.data_parts) == 4
    assert sum(len(part.data) for part in test_dataset.data_parts) == 6


# Tests for split_by_dates function
def test_split_by_dates__splits_by_multiple_dates(sample_daily_dataset: TimeSeriesDataset):
    """Test that split_by_dates splits data based on multiple dates."""
    # Arrange
    test_dates = pd.DatetimeIndex(["2023-01-03", "2023-01-07"])

    # Act
    train_dataset, test_dataset = split_by_dates(sample_daily_dataset, test_dates)

    # Assert
    # Check that train contains all dates EXCEPT the test dates
    expected_train_dates = sample_daily_dataset.index.difference(test_dates)  # type: ignore
    assert train_dataset.index.to_series().equals(expected_train_dates.to_series())

    # Check that test contains ONLY the specified dates
    assert test_dataset.index.to_series().equals(test_dates.to_series())


def test_split_by_dates__empty_dates(sample_daily_dataset: TimeSeriesDataset):
    """Test split_by_dates with empty date list."""
    # Arrange
    test_dates = pd.DatetimeIndex([])

    # Act
    train_dataset, test_dataset = split_by_dates(sample_daily_dataset, test_dates)

    # Assert
    assert len(train_dataset.data) == 10
    assert len(test_dataset.data) == 0


# Tests for ChronologicalTrainTestSplitter
def test_chronological_train_test_splitter__initialization():
    """Test ChronologicalTrainTestSplitter initialization."""
    # Arrange
    # Act
    splitter = ChronologicalTrainTestSplitter(test_fraction=0.2)

    # Assert
    assert splitter.test_fraction == 0.2
    assert not splitter.is_fitted


def test_chronological_train_test_splitter__fit(sample_daily_dataset: TimeSeriesDataset):
    """Test ChronologicalTrainTestSplitter fit method."""
    # Arrange
    splitter = ChronologicalTrainTestSplitter(test_fraction=0.2)

    # Act
    splitter.fit(sample_daily_dataset)

    # Assert
    assert splitter.is_fitted
    assert splitter._split_date is not None
    # With 10 samples and test_fraction=0.2, split at index 8
    assert splitter._split_date == sample_daily_dataset.index[8]


def test_chronological_train_test_splitter__fit_invalid_fraction(sample_daily_dataset: TimeSeriesDataset):
    """Test ChronologicalTrainTestSplitter raises error for invalid test_fraction."""
    # Arrange
    # Act
    # Assert
    with pytest.raises(pydantic.ValidationError):
        ChronologicalTrainTestSplitter(test_fraction=1.5)


def test_chronological_train_test_splitter__transform_not_fitted(sample_daily_dataset: TimeSeriesDataset):
    """Test ChronologicalTrainTestSplitter raises error when transforming before fitting."""
    # Arrange
    splitter = ChronologicalTrainTestSplitter()

    # Act
    # Assert
    with pytest.raises(NotFittedError):
        splitter.transform(sample_daily_dataset)


def test_chronological_train_test_splitter__fit_transform(sample_daily_dataset: TimeSeriesDataset):
    """Test ChronologicalTrainTestSplitter fit and transform workflow."""
    # Arrange
    splitter = ChronologicalTrainTestSplitter(test_fraction=0.2)

    # Act
    splitter.fit(sample_daily_dataset)
    train_dataset, test_dataset = splitter.transform(sample_daily_dataset)

    # Assert
    # With 10 samples and test_fraction=0.2, expect 8 train, 2 test
    assert len(train_dataset.data) == 8
    assert len(test_dataset.data) == 2

    # Check that concatenated indexes equal original (verifies no data loss and correct split)
    combined_index = train_dataset.index.append(test_dataset.index)
    assert combined_index.to_series().equals(sample_daily_dataset.index.to_series())


# Tests for StratifiedTrainTestSplitter
def test_stratified_train_test_splitter__initialization():
    """Test StratifiedTrainTestSplitter initialization."""
    # Arrange
    # Act
    splitter = StratifiedTrainTestSplitter(
        test_fraction=0.2,
        stratification_fraction=0.15,
        target_column="load",
        random_state=42,
        min_days_for_stratification=4,
    )

    # Assert
    assert splitter.test_fraction == 0.2
    assert splitter.stratification_fraction == 0.15
    assert splitter.target_column == "load"
    assert splitter.random_state == 42
    assert splitter.min_days_for_stratification == 4
    assert not splitter.is_fitted


def test_stratified_train_test_splitter__fallback_to_chronological(sample_daily_dataset: TimeSeriesDataset):
    """Test StratifiedTrainTestSplitter falls back to chronological with insufficient days."""
    # Arrange
    # Create dataset with only 3 days (less than min_days_for_stratification=4)
    short_data = sample_daily_dataset.data.iloc[:3]
    short_dataset = TimeSeriesDataset(data=short_data, sample_interval=timedelta(days=1))
    splitter = StratifiedTrainTestSplitter(min_days_for_stratification=4)

    # Act
    splitter.fit(short_dataset)

    # Assert
    assert splitter.is_fitted
    assert splitter._chronological_splitter is not None
    assert splitter._test_dates is None


def test_stratified_train_test_splitter__fit_with_sufficient_days(stratified_daily_dataset: TimeSeriesDataset):
    """Test StratifiedTrainTestSplitter fit with sufficient days for stratification."""
    # Arrange
    splitter = StratifiedTrainTestSplitter(test_fraction=0.2, stratification_fraction=0.15, random_state=42)

    # Act
    splitter.fit(stratified_daily_dataset)

    # Assert
    assert splitter.is_fitted
    assert splitter._test_dates is not None
    assert splitter._chronological_splitter is None

    # Check that test dates are a subset of the original dataset dates
    assert splitter._test_dates.isin(stratified_daily_dataset.index).all()  # type: ignore

    # Check that we have the expected number of test dates (approximately 20% of 20 = 4, but stratified logic may vary)
    assert 3 <= len(splitter._test_dates) <= 5

    # Check that test dates include at least one extreme day from min and max groups
    target_series = stratified_daily_dataset.data["load"]
    max_days, min_days, _ = StratifiedTrainTestSplitter._get_extreme_days(target_series=target_series, fraction=0.15)
    assert not splitter._test_dates.intersection(max_days).empty, (
        "Test dates should include at least one max extreme day"
    )
    assert not splitter._test_dates.intersection(min_days).empty, (
        "Test dates should include at least one min extreme day"
    )


def test_stratified_train_test_splitter__transform_not_fitted(stratified_daily_dataset: TimeSeriesDataset):
    """Test StratifiedTrainTestSplitter raises error when transforming before fitting."""
    # Arrange
    splitter = StratifiedTrainTestSplitter()

    # Act
    # Assert
    with pytest.raises(NotFittedError):
        splitter.transform(stratified_daily_dataset)


def test_stratified_train_test_splitter__fit_transform(stratified_daily_dataset: TimeSeriesDataset):
    """Test StratifiedTrainTestSplitter fit and transform workflow."""
    # Arrange
    splitter = StratifiedTrainTestSplitter(test_fraction=0.2, stratification_fraction=0.15, random_state=42)

    # Act
    splitter.fit(stratified_daily_dataset)
    train_dataset, test_dataset = splitter.transform(stratified_daily_dataset)

    # Assert
    # Check data integrity - combined data should equal original (when sorted by index)
    combined_data = pd.concat([train_dataset.data, test_dataset.data]).sort_index()
    assert combined_data.equals(stratified_daily_dataset.data)

    # Check that train and test indexes are disjoint (no overlap)
    assert train_dataset.index.intersection(test_dataset.index).empty


def test_stratified_train_test_splitter__fallback_transform(sample_daily_dataset: TimeSeriesDataset):
    """Test StratifiedTrainTestSplitter fallback transform behavior."""
    # Arrange
    # Create dataset with insufficient days
    short_data = sample_daily_dataset.data.iloc[:3]
    short_dataset = TimeSeriesDataset(data=short_data, sample_interval=timedelta(days=1))
    splitter = StratifiedTrainTestSplitter(min_days_for_stratification=4)

    # Act
    splitter.fit(short_dataset)
    train_dataset, test_dataset = splitter.transform(short_dataset)

    # Assert
    # Should behave like chronological split
    assert len(train_dataset.data) == 2  # 3 total, test_fraction=0.2 -> 2 train, 1 test
    assert len(test_dataset.data) == 1
    # Check that concatenated indexes equal original (verifies no data loss and correct split)
    combined_index = train_dataset.index.append(test_dataset.index)
    assert combined_index.to_series().equals(short_dataset.index.to_series())


# Tests for helper methods
def test_stratified_train_test_splitter__get_extreme_days_deterministic(stratified_daily_dataset: TimeSeriesDataset):
    """Test _get_extreme_days with deterministic data to verify correct extreme selection."""
    # Arrange
    target_series = stratified_daily_dataset.data["load"]

    # Act
    max_days, min_days, other_days = StratifiedTrainTestSplitter._get_extreme_days(
        target_series=target_series, fraction=0.2
    )

    # Assert
    # With 20 days and fraction=0.2, expect 4 extreme days each (2 min + 2 max = 4 total)
    assert len(max_days) == 4
    assert len(min_days) == 4
    assert len(other_days) == 12

    # Check specific dates are selected as extremes (determined dynamically from fixture)
    expected_min_dates = stratified_daily_dataset.index[:4]  # First 4 days (lowest values)
    expected_max_dates = stratified_daily_dataset.index[-4:][::-1]  # Last 4 days in descending order (highest values)

    assert max_days.to_series().equals(expected_max_dates.to_series())
    assert min_days.to_series().equals(expected_min_dates.to_series())

    # Check no overlap
    assert len(max_days.intersection(min_days)) == 0
    assert len(max_days.intersection(other_days)) == 0
    assert len(min_days.intersection(other_days)) == 0


@pytest.mark.parametrize(
    ("dates", "test_fraction"),
    [
        (pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]), 0.25),
        (pd.DatetimeIndex(["2023-01-01"]), 0.5),
        (pd.DatetimeIndex([]), 0.2),
    ],
)
def test_stratified_train_test_splitter__sample_dates_for_split(dates: pd.DatetimeIndex, test_fraction: float):
    """Test _sample_dates_for_split helper method."""
    # Arrange
    rng = np.random.default_rng(42)

    # Act
    train_dates, test_dates = StratifiedTrainTestSplitter._sample_dates_for_split(
        dates=dates, test_fraction=test_fraction, rng=rng
    )

    # Assert
    # For non-empty dates, check basic properties
    if not dates.empty:
        assert len(train_dates) + len(test_dates) == len(dates)
        assert len(set(train_dates).intersection(set(test_dates))) == 0
        # All dates should be accounted for
        assert set(train_dates).union(set(test_dates)) == set(dates)
    else:
        assert len(train_dates) == 0
        assert len(test_dates) == 0
