# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError, NotFittedError
from openstef_core.testing import create_synthetic_forecasting_dataset
from openstef_models.transforms.validation import InputConsistencyChecker


@pytest.fixture
def base_dataset() -> TimeSeriesDataset:
    """Fixture providing a base synthetic forecasting dataset."""
    return create_synthetic_forecasting_dataset(
        wind_influence=1.0,
        temp_influence=1.0,
        radiation_influence=None,
        stochastic_influence=None,
    )


def test_transform_successful_case(base_dataset: TimeSeriesDataset):
    """Test successful transformation with matching columns."""
    # Arrange
    checker = InputConsistencyChecker()

    # Act
    checker.fit(base_dataset)
    result = checker.transform(base_dataset)

    # Assert
    pd.testing.assert_frame_equal(result.data, base_dataset.data)


def test_transform_missing_column_raises_error(base_dataset: TimeSeriesDataset):
    """Test transformation raises error when required column is missing."""
    # Arrange
    # Create transform dataset with one column removed
    transform_dataset = base_dataset.select_features(["load", "windspeed"])
    # Note: temperature column is missing

    checker = InputConsistencyChecker()
    checker.fit(base_dataset)

    # Act & Assert
    with pytest.raises(MissingColumnsError, match="Missing required columns: temperature"):
        checker.transform(transform_dataset)


def test_transform_extra_column_logs_warning(caplog: LogCaptureFixture, base_dataset: TimeSeriesDataset):
    """Test transformation logs warning for extra columns but continues."""
    # Arrange
    # Create transform dataset with extra column by setting radiation_influence
    fit_dataset = base_dataset.select_features(["load", "windspeed"])
    transform_dataset = base_dataset

    checker = InputConsistencyChecker()
    checker.fit(fit_dataset)

    # Act
    result = checker.transform(transform_dataset)

    # Assert
    assert "Input data contains extra columns not seen during fitting: {'temperature'}" in caplog.text
    # Should still return the data but with extra columns removed
    assert list(result.data.columns) == ["load", "windspeed"]


def test_transform_not_fitted_raises_error(base_dataset: TimeSeriesDataset):
    """Test transformation raises error when checker is not fitted."""
    # Arrange
    checker = InputConsistencyChecker()

    # Act & Assert
    with pytest.raises(NotFittedError, match="The InputConsistencyChecker has not been fitted yet"):
        checker.transform(base_dataset)
