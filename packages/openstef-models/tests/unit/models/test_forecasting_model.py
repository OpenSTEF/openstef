# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting import BaseForecaster, BaseHorizonForecaster
from openstef_models.models.forecasting.mixins import ForecasterConfig, HorizonForecasterConfig
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms import FeaturePipeline, PostprocessingPipeline


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    # Create sample time series data with typical energy forecasting features
    data = pd.DataFrame(
        {
            "load": [100.0, 110.0, 120.0, 105.0, 95.0, 115.0],
            "temperature": [20.0, 21.0, 22.0, 20.5, 19.5, 21.5],
            "radiation": [0.0, 100.0, 500.0, 300.0, 50.0, 250.0],
        },
        index=pd.date_range("2025-01-01 10:00", periods=6, freq="h"),
    )

    return TimeSeriesDataset(data, timedelta(hours=1))


def test_forecasting_model__init__validates_horizons_and_assigns_components():
    """Test initialization validates horizon matching and assigns components correctly."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = horizons

    mock_postprocessing = MagicMock(spec=PostprocessingPipeline)

    # Act
    model = ForecastingModel(
        forecaster=mock_forecaster,
        preprocessing=mock_preprocessing,
        postprocessing=mock_postprocessing,
        target_column="energy",
    )

    # Assert - Components are assigned correctly
    assert model.forecaster is mock_forecaster
    assert model.preprocessing is mock_preprocessing
    assert model.postprocessing is mock_postprocessing
    assert model.target_column == "energy"
    assert model.is_fitted == mock_forecaster.is_fitted  # Property delegation


def test_forecasting_model__init__uses_defaults_and_validates_horizons():
    """Test initialization with defaults and horizon validation errors."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
    )

    # Act & Assert - Test defaults work
    model = ForecastingModel(forecaster=mock_forecaster)

    # Check that default FeaturePipeline and PostprocessingPipeline were created
    assert model.preprocessing is not None
    assert isinstance(model.preprocessing, FeaturePipeline)
    assert model.preprocessing.horizons == mock_forecaster.config.horizons
    assert model.postprocessing is not None
    assert isinstance(model.postprocessing, PostprocessingPipeline)
    assert model.target_column == "load"  # Default value

    # Act & Assert - Test horizon validation error
    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = [LeadTime(timedelta(hours=6))]  # Different horizons

    with pytest.raises(ConfigurationError, match=r"forecaster horizons.*do not match.*preprocessing horizons"):
        ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)


@pytest.mark.parametrize(
    "forecaster_type",
    [
        pytest.param(BaseForecaster, id="base_forecaster"),
        pytest.param(BaseHorizonForecaster, id="horizon_forecaster"),
    ],
)
def test_forecasting_model__fit_and_predict__orchestrates_correctly(
    forecaster_type: type, sample_timeseries_dataset: TimeSeriesDataset
):
    """Test that fit and predict correctly orchestrate preprocessing and forecaster calls."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    mock_forecaster = MagicMock(spec=forecaster_type)

    if forecaster_type == BaseForecaster:
        mock_forecaster.config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    else:
        mock_forecaster.config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)

    mock_forecaster.is_fitted = False

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = horizons

    mock_postprocessing = MagicMock(spec=PostprocessingPipeline)

    model = ForecastingModel(
        forecaster=mock_forecaster,
        preprocessing=mock_preprocessing,
        postprocessing=mock_postprocessing,
    )

    # Set up preprocessing mock to return real TimeSeriesDataset for _prepare_input_data
    mock_preprocessing.transform.return_value = {LeadTime(timedelta(hours=6)): sample_timeseries_dataset}

    # Act - Test fit workflow
    model.fit(dataset=sample_timeseries_dataset)

    # Assert - Fit orchestration
    mock_preprocessing.fit.assert_called_once_with(dataset=sample_timeseries_dataset)
    mock_preprocessing.transform.assert_called_once_with(dataset=sample_timeseries_dataset)

    if forecaster_type == BaseForecaster:
        # Should be called with dict containing ForecastInputDataset
        mock_forecaster.fit.assert_called_once()
        fit_call_args = mock_forecaster.fit.call_args[1]["input_data"]
        assert len(fit_call_args) == 1
        assert LeadTime(timedelta(hours=6)) in fit_call_args
        assert isinstance(fit_call_args[LeadTime(timedelta(hours=6))], ForecastInputDataset)
    else:
        # HorizonForecaster gets first data item
        mock_forecaster.fit_horizon.assert_called_once()
        fit_call_args = mock_forecaster.fit_horizon.call_args[1]["input_data"]
        assert isinstance(fit_call_args, ForecastInputDataset)

    # Arrange for predict - Set fitted and add return values
    mock_forecaster.is_fitted = True
    mock_raw_forecast = MagicMock(spec=ForecastDataset)
    final_forecast = MagicMock(spec=ForecastDataset)
    mock_postprocessing.transform.return_value = final_forecast

    if forecaster_type == BaseForecaster:
        mock_forecaster.predict.return_value = mock_raw_forecast
    else:
        mock_forecaster.predict_horizon.return_value = mock_raw_forecast

    forecast_start = datetime.fromisoformat("2025-01-01T12:00:00")

    # Act - Test predict workflow
    result = model.predict(dataset=sample_timeseries_dataset, forecast_start=forecast_start)

    # Assert - Predict orchestration
    if forecaster_type == BaseForecaster:
        mock_forecaster.predict.assert_called_once()
        predict_call_args = mock_forecaster.predict.call_args[1]["input_data"]
        assert len(predict_call_args) == 1
        assert LeadTime(timedelta(hours=6)) in predict_call_args
        assert isinstance(predict_call_args[LeadTime(timedelta(hours=6))], ForecastInputDataset)
    else:
        mock_forecaster.predict_horizon.assert_called_once()
        predict_call_args = mock_forecaster.predict_horizon.call_args[1]["input_data"]
        assert isinstance(predict_call_args, ForecastInputDataset)

    mock_postprocessing.transform.assert_called_once_with(mock_raw_forecast)
    assert result is final_forecast


def test_forecasting_model__error_conditions(sample_timeseries_dataset: TimeSeriesDataset):
    """Test error conditions for fit and predict operations."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    mock_forecaster = MagicMock(spec=BaseHorizonForecaster)
    mock_forecaster.config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    mock_forecaster.is_fitted = False

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = horizons

    model = ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)

    # Test predict when not fitted
    with pytest.raises(NotFittedError):
        model.predict(dataset=sample_timeseries_dataset)

    # Test fit with no horizon data - set preprocessing to return empty dict
    mock_preprocessing.transform.return_value = {}
    with pytest.raises(ValueError, match="No data available for horizon forecasting"):
        model.fit(dataset=sample_timeseries_dataset)
