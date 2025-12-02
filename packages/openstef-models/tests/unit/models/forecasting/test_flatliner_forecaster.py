# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.flatliner_forecaster import FlatlinerForecaster, FlatlinerForecasterConfig


@pytest.fixture
def config() -> FlatlinerForecasterConfig:
    return FlatlinerForecasterConfig(
        quantiles=[Quantile(0.5), Quantile(0.9)], horizons=[LeadTime(timedelta(hours=1)), LeadTime(timedelta(hours=2))]
    )


def test_predict_returns_zeros(config: FlatlinerForecasterConfig, sample_forecast_input_dataset: ForecastInputDataset):
    forecaster = FlatlinerForecaster(config)
    result = forecaster.predict(sample_forecast_input_dataset)
    assert isinstance(result.data, pd.DataFrame)
    assert (result.data == 0.0).all().all()
    assert set(result.data.columns) == {q.format() for q in config.quantiles}


def test_is_fitted_always_true(config: FlatlinerForecasterConfig):
    forecaster = FlatlinerForecaster(config)
    assert forecaster.is_fitted


def test_predict_returns_median_when_use_median_is_true(sample_forecast_input_dataset: ForecastInputDataset):
    """Test that the forecaster predicts the median of load measurements when use_median is True."""
    # Arrange
    config = FlatlinerForecasterConfig(
        quantiles=[Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime(timedelta(hours=1))],
        use_median=True,
    )
    forecaster = FlatlinerForecaster(config)

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    expected_median = sample_forecast_input_dataset.target_series.median()
    assert forecaster.is_fitted
    assert isinstance(result.data, pd.DataFrame)
    assert (result.data == expected_median).all().all()
    assert set(result.data.columns) == {q.format() for q in config.quantiles}
