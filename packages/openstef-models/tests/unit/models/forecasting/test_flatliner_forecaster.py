# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.flatliner_forecaster import FlatlinerForecaster
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow


@pytest.fixture
def config() -> FlatlinerForecaster:
    return FlatlinerForecaster(
        quantiles=[Quantile(0.5), Quantile(0.9)], horizons=[LeadTime(timedelta(hours=1)), LeadTime(timedelta(hours=2))]
    )


def test_predict_returns_zeros(config: FlatlinerForecaster, sample_forecast_input_dataset: ForecastInputDataset):
    forecaster = config
    result = forecaster.predict(sample_forecast_input_dataset)
    assert isinstance(result.data, pd.DataFrame)
    assert (result.data == 0.0).all().all()
    assert set(result.data.columns) == {q.format() for q in config.quantiles}


def test_is_fitted_always_true(config: FlatlinerForecaster):
    forecaster = config
    assert forecaster.is_fitted


def test_feature_importances_shape_matches_quantiles():
    """Feature importances must have one column per quantile."""
    quantiles = [Quantile(0.1), Quantile(0.3), Quantile(0.5), Quantile(0.7), Quantile(0.9)]
    forecaster = FlatlinerForecaster(quantiles=quantiles, horizons=[LeadTime(timedelta(hours=1))])

    importances = forecaster.feature_importances

    assert isinstance(importances, pd.DataFrame)
    assert list(importances.columns) == [q.format() for q in quantiles]
    assert list(importances.index) == ["load"]
    assert (importances == 0.0).all().all()  # NOSONAR python:S1244 - flatliner returns exact 0.0


def test_predict_returns_median_when_predict_median_is_true(sample_forecast_input_dataset: ForecastInputDataset):
    """Test that the forecaster predicts the median of load measurements when predict_median is True."""
    # Arrange
    forecaster = FlatlinerForecaster(
        quantiles=[Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime(timedelta(hours=1))],
        predict_median=True,
    )

    # Act
    forecaster.fit(sample_forecast_input_dataset)
    result = forecaster.predict(sample_forecast_input_dataset)

    # Assert
    expected_median = sample_forecast_input_dataset.target_series.median()
    assert forecaster.is_fitted
    assert isinstance(result.data, pd.DataFrame)
    assert (result.data == expected_median).all().all()
    assert set(result.data.columns) == {q.format() for q in forecaster.quantiles}


@pytest.mark.parametrize(
    ("load_values", "median_window", "expected_median"),
    [
        pytest.param([100.0, 110.0, 120.0, 7.0, 7.0, 7.0], timedelta(hours=2), 7.0, id="repeated-value"),
        pytest.param([100.0, 110.0, 120.0, 6.0, 8.0], timedelta(hours=1), 7.0, id="two-different-values"),
    ],
)
def test_predict_returns_recent_median_when_median_window_is_set(
    load_values: list[float],
    median_window: timedelta,
    expected_median: float,
) -> None:
    # Arrange
    index = pd.date_range("2025-01-01", periods=len(load_values), freq="h")
    data = ForecastInputDataset(
        data=pd.DataFrame({"load": load_values}, index=index),
        sample_interval=timedelta(hours=1),
    )
    forecaster = FlatlinerForecaster(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
        predict_median=True,
        median_window=median_window,
    )

    # Act
    forecaster.fit(data)
    result = forecaster.predict(data)

    # Assert
    assert (result.data == expected_median).all().all()


def test_flatliner_workflow_uses_threshold_for_median_window() -> None:
    # Arrange
    config = ForecastingWorkflowConfig(
        model_id="flatliner-test",
        model="flatliner",
        flatliner_threshold=timedelta(hours=6),
        predict_nonzero_flatliner=True,
        mlflow_storage=None,
    )

    # Act
    workflow = create_forecasting_workflow(config)

    # Assert
    assert isinstance(workflow.model, ForecastingModel)
    assert isinstance(workflow.model.forecaster, FlatlinerForecaster)
    assert workflow.model.forecaster.median_window == timedelta(hours=6)
