# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
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
