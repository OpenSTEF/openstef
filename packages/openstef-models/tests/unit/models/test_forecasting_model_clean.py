# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting import BaseForecaster, BaseHorizonForecaster
from openstef_models.models.forecasting.mixins import ForecasterConfig, HorizonForecasterConfig
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms import FeaturePipeline, PostprocessingPipeline


def test_forecasting_model__init__assigns_components_correctly():
    """Test that ForecastingModel correctly assigns all components during initialization."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    mock_postprocessing = MagicMock(spec=PostprocessingPipeline)

    # Act
    model = ForecastingModel(
        forecaster=mock_forecaster,
        preprocessing=mock_preprocessing,
        postprocessing=mock_postprocessing,
        target_column="energy",
    )

    # Assert
    assert model.forecaster is mock_forecaster
    assert model.preprocessing is mock_preprocessing
    assert model.postprocessing is mock_postprocessing
    assert model.target_column == "energy"


def test_forecasting_model__init__uses_defaults_when_none():
    """Test that ForecastingModel uses default pipelines when None is provided."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1))],
    )

    # Act
    with (
        patch("openstef_models.models.forecasting_model.FeaturePipeline") as mock_feature_cls,
        patch("openstef_models.models.forecasting_model.PostprocessingPipeline") as mock_postproc_cls,
    ):
        mock_feature_instance = MagicMock()
        mock_feature_instance.horizons = mock_forecaster.config.horizons
        mock_feature_cls.return_value = mock_feature_instance

        model = ForecastingModel(forecaster=mock_forecaster)

        # Assert
        mock_feature_cls.assert_called_once()
        mock_postproc_cls.assert_called_once()
        assert model.target_column == "load"  # Default value


def test_forecasting_model__init__raises_error_mismatched_horizons():
    """Test that initialization raises ConfigurationError when horizons don't match."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1)), LeadTime(timedelta(hours=6))],
    )

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = [LeadTime(timedelta(hours=1))]  # Different horizons

    # Act & Assert
    with pytest.raises(ConfigurationError, match=r"forecaster horizons.*do not match.*preprocessing horizons"):
        ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)


def test_forecasting_model__is_fitted__reflects_forecaster_state():
    """Test that is_fitted property correctly reflects forecaster state."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )
    mock_forecaster.is_fitted = True

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    model = ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)

    # Act & Assert
    assert model.is_fitted is True


def test_forecasting_model__fit__calls_preprocessing_and_forecaster_correctly():
    """Test that fit() correctly orchestrates preprocessing and forecaster training."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    sample_dataset = MagicMock(spec=TimeSeriesDataset)

    model = ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)

    # Act
    with patch.object(model, "_prepare_input_data") as mock_prepare:
        mock_input_data = {LeadTime(timedelta(hours=6)): MagicMock(spec=ForecastInputDataset)}
        mock_prepare.return_value = mock_input_data

        model.fit(dataset=sample_dataset)

    # Assert
    mock_preprocessing.fit.assert_called_once_with(dataset=sample_dataset)
    mock_prepare.assert_called_once_with(dataset=sample_dataset)
    mock_forecaster.fit.assert_called_once_with(input_data=mock_input_data)


def test_forecasting_model__fit__handles_horizon_forecaster_correctly():
    """Test that fit() handles BaseHorizonForecaster correctly."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseHorizonForecaster)
    mock_forecaster.config = HorizonForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    sample_dataset = MagicMock(spec=TimeSeriesDataset)

    model = ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)

    # Act
    with patch.object(model, "_prepare_input_data") as mock_prepare:
        mock_horizon_data = MagicMock(spec=ForecastInputDataset)
        mock_input_data = {LeadTime(timedelta(hours=6)): mock_horizon_data}
        mock_prepare.return_value = mock_input_data

        model.fit(dataset=sample_dataset)

    # Assert
    mock_forecaster.fit_horizon.assert_called_once_with(input_data=mock_horizon_data)


def test_forecasting_model__fit__raises_error_no_horizon_data():
    """Test that fit() raises ValueError when no data is available for horizon forecasting."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseHorizonForecaster)
    mock_forecaster.config = HorizonForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    model = ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)

    # Act & Assert
    with (
        patch.object(model, "_prepare_input_data", return_value={}),
        pytest.raises(ValueError, match="No data available for horizon forecasting"),
    ):
        model.fit(dataset=MagicMock())


def test_forecasting_model__predict__raises_error_when_not_fitted():
    """Test that predict() raises ModelNotFittedError when model is not fitted."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )
    mock_forecaster.is_fitted = False

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    model = ForecastingModel(forecaster=mock_forecaster, preprocessing=mock_preprocessing)

    # Act & Assert
    with pytest.raises(NotFittedError):
        model.predict(dataset=MagicMock())


def test_forecasting_model__predict__orchestrates_prediction_correctly():
    """Test that predict() correctly orchestrates the full prediction pipeline."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )
    mock_forecaster.is_fitted = True

    mock_raw_forecast = MagicMock(spec=ForecastDataset)
    mock_forecaster.predict.return_value = mock_raw_forecast

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    mock_postprocessing = MagicMock(spec=PostprocessingPipeline)
    final_forecast = MagicMock(spec=ForecastDataset)
    mock_postprocessing.transform.return_value = final_forecast

    model = ForecastingModel(
        forecaster=mock_forecaster,
        preprocessing=mock_preprocessing,
        postprocessing=mock_postprocessing,
    )

    sample_dataset = MagicMock(spec=TimeSeriesDataset)
    forecast_start = datetime.fromisoformat("2025-01-01T03:00:00")

    # Act
    with patch.object(model, "_prepare_input_data") as mock_prepare:
        mock_input_data = {LeadTime(timedelta(hours=6)): MagicMock(spec=ForecastInputDataset)}
        mock_prepare.return_value = mock_input_data

        result = model.predict(dataset=sample_dataset, forecast_start=forecast_start)

    # Assert
    mock_prepare.assert_called_once_with(dataset=sample_dataset, forecast_start=forecast_start)
    mock_forecaster.predict.assert_called_once_with(input_data=mock_input_data)
    mock_postprocessing.transform.assert_called_once_with(mock_raw_forecast)
    assert result is final_forecast


def test_forecasting_model__prepare_input_data__transforms_data_correctly():
    """Test that _prepare_input_data correctly transforms input data."""
    # Arrange
    mock_forecaster = MagicMock(spec=BaseForecaster)
    mock_forecaster.config = ForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=1)), LeadTime(timedelta(hours=6))],
    )

    mock_preprocessing = MagicMock(spec=FeaturePipeline)
    mock_preprocessing.horizons = mock_forecaster.config.horizons

    sample_dataset = MagicMock(spec=TimeSeriesDataset)
    mock_preprocessing.transform.return_value = {
        LeadTime(timedelta(hours=1)): sample_dataset,
        LeadTime(timedelta(hours=6)): sample_dataset,
    }

    model = ForecastingModel(
        forecaster=mock_forecaster,
        preprocessing=mock_preprocessing,
        target_column="energy",
    )

    forecast_start = datetime.fromisoformat("2025-01-01T03:00:00")

    # Act
    with patch("openstef_models.models.forecasting_model.ForecastInputDataset") as mock_forecast_input:
        mock_forecast_input.from_timeseries_dataset.return_value = MagicMock(spec=ForecastInputDataset)

        result = model._prepare_input_data(dataset=sample_dataset, forecast_start=forecast_start)

    # Assert
    # Check preprocessing transform was called
    mock_preprocessing.transform.assert_called_once_with(dataset=sample_dataset)

    # Check ForecastInputDataset creation for each horizon
    assert mock_forecast_input.from_timeseries_dataset.call_count == 2
    call_args_list = mock_forecast_input.from_timeseries_dataset.call_args_list

    for call_args in call_args_list:
        _, kwargs = call_args
        assert kwargs["dataset"] is sample_dataset
        assert kwargs["target_column"] == "energy"
        assert kwargs["forecast_start"] == forecast_start

    # Check result structure
    assert len(result) == 2
    assert LeadTime(timedelta(hours=1)) in result
    assert LeadTime(timedelta(hours=6)) in result
