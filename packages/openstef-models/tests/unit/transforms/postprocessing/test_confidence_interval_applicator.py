# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for ConfidenceIntervalApplicator transform."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import Quantile
from openstef_models.transforms.postprocessing.confidence_interval_applicator import (
    ConfidenceIntervalApplicator,
)


@pytest.fixture
def validation_predictions() -> ForecastDataset:
    """Validation predictions spanning 6 days."""
    index = pd.date_range("2018-01-01 00:00:00", periods=144, freq="1h")

    return ForecastDataset(
        data=pd.DataFrame(
            {
                "load": np.tile([4.0, 2.0, 5.0, 2.0, 4.0, 2.0, 5.0, 2.0], 18),
                Quantile(0.5).format(): np.tile([5.0, 6.0, 7.0, 8.0], 36),
                "horizon": [timedelta(hours=6)] * 72 + [timedelta(hours=12)] * 72,
            },
            index=index,
        ),
        sample_interval=timedelta(minutes=15),
        forecast_start=datetime.fromisoformat("2018-01-01T00:00:00"),
    )


@pytest.fixture
def predictions() -> ForecastDataset:
    """Predictions to be processed."""
    forecast_start = datetime.fromisoformat("2018-01-03T00:00:00")
    forecast_index = pd.date_range(forecast_start, periods=10, freq="1h")
    return ForecastDataset(
        data=pd.DataFrame(
            {
                "load": [4.0, 2.0, 5.0, 2.0, 4.0, 2.0, 5.0, 2.0, 4.0, 2.0],
                Quantile(0.5).format(): np.ones(10) * 10.0,
                "horizon": timedelta(days=6),
            },
            index=forecast_index,
        ),
        sample_interval=timedelta(minutes=15),
        forecast_start=forecast_start,
    )


def test_single_horizon_workflow(
    validation_predictions: ForecastDataset,
    predictions: ForecastDataset,
):
    """Test complete single-horizon workflow with quantile generation."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]
    applicator = ConfidenceIntervalApplicator(quantiles=quantiles)

    # Act
    applicator.fit(validation_predictions)
    result = applicator.transform(predictions)

    # Assert
    # Should have all quantile columns
    for q in quantiles:
        assert q.format() in result.data.columns

    # Quantiles should follow normal distribution properties: P10 <= P50 <= P90
    assert (result.data["quantile_P10"] <= result.data["quantile_P50"]).all()
    assert (result.data["quantile_P50"] <= result.data["quantile_P90"]).all()

    # P50 should match original median (within floating point tolerance)
    pd.testing.assert_series_equal(
        result.data["quantile_P50"],
        predictions.data["quantile_P50"],
        rtol=1e-10,
    )

    # No NaN values
    assert not result.data["quantile_P10"].isna().any()
    assert not result.data["quantile_P50"].isna().any()
    assert not result.data["quantile_P90"].isna().any()


def test_multi_horizon_workflow(
    validation_predictions: ForecastDataset,
    predictions: ForecastDataset,
):
    """Test complete multi-horizon workflow."""
    # Arrange
    applicator = ConfidenceIntervalApplicator(quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)])

    # Act
    applicator.fit(validation_predictions)
    result = applicator.transform(predictions)

    # Assert
    # Should learn stdev for both horizons
    assert applicator.standard_deviation.shape == (24, 2)
    assert "PT6H" in applicator.standard_deviation.columns
    assert "PT12H" in applicator.standard_deviation.columns

    # Should have quantile columns
    assert "quantile_P10" in result.data.columns
    assert "quantile_P50" in result.data.columns
    assert "quantile_P90" in result.data.columns

    # Quantiles should follow normal distribution properties
    assert (result.data["quantile_P10"] <= result.data["quantile_P50"]).all()
    assert (result.data["quantile_P50"] <= result.data["quantile_P90"]).all()

    # P50 should match original median
    pd.testing.assert_series_equal(
        result.data["quantile_P50"],
        predictions.data["quantile_P50"],
        rtol=1e-10,
    )

    # Uncertainty (quantile spread) should increase with forecast horizon
    spread = result.data["quantile_P90"] - result.data["quantile_P10"]
    assert spread.iloc[5:].mean() >= spread.iloc[:5].mean()


def test_fit__computes_hourly_standard_deviation_correctly(
    validation_predictions: ForecastDataset,
):
    """Test that fit() correctly computes hour-specific standard deviations."""
    # Arrange
    applicator = ConfidenceIntervalApplicator()

    # Act
    applicator.fit(validation_predictions)

    # Assert
    stdev_df = applicator.standard_deviation
    assert stdev_df.index.name == "hour"
    assert len(stdev_df) == 24
    assert stdev_df.shape[1] == 2  # Two horizons: 6h and 12h
    assert (stdev_df >= 0).all().all()
    assert not stdev_df.isna().any().any()


def test_transform__raises_error_when_not_fitted(
    predictions: ForecastDataset,
):
    """Test that transform() raises NotFittedError when called before fit()."""
    # Arrange
    applicator = ConfidenceIntervalApplicator(quantiles=[Quantile(0.5)])

    # Act & Assert
    with pytest.raises(NotFittedError, match="ConfidenceIntervalApplicator"):
        applicator.transform(predictions)


def test_state_roundtrip(
    validation_predictions: ForecastDataset,
):
    """Test state serialization and restoration."""
    # Arrange
    original_transform = ConfidenceIntervalApplicator(quantiles=[Quantile(0.1), Quantile(0.9)])
    original_transform.fit(validation_predictions)

    # Act
    state = original_transform.to_state()
    restored_transform = ConfidenceIntervalApplicator(quantiles=[Quantile(0.1), Quantile(0.9)])
    restored_transform = restored_transform.from_state(state)

    # Assert
    pd.testing.assert_frame_equal(original_transform.standard_deviation, restored_transform.standard_deviation)
