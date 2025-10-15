# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import Self, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.mixins import State, TransformPipeline
from openstef_core.types import LeadTime, Quantile, override
from openstef_models.models.forecasting import Forecaster, HorizonForecaster, HorizonForecasterConfig
from openstef_models.models.forecasting.multi_horizon_forecaster_adapter import (
    MultiHorizonForecasterAdapter,
    MultiHorizonForecasterConfig,
)
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms import FeatureEngineeringPipeline


class SimpleForecaster(HorizonForecaster):
    """Simple test forecaster that returns predictable values for testing."""

    def __init__(self, config: HorizonForecasterConfig):
        self._config = config
        self._is_fitted = False

    @property
    def config(self) -> HorizonForecasterConfig:
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def to_state(self) -> State:
        return cast(State, self)

    @override
    def from_state(self, state: State) -> Self:  # noqa: PLR6301
        return cast(Self, state)

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        self._is_fitted = True

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        # Return predictable forecast values
        forecast_values = {quantile: 100.0 + quantile * 10 for quantile in self.config.quantiles}
        return ForecastDataset(
            pd.DataFrame(
                {
                    quantile.format(): [forecast_values[quantile]] * len(data.index)
                    for quantile in self.config.quantiles
                },
                index=data.index,
            ),
            data.sample_interval,
            data.forecast_start,
        )


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample time series data with typical energy forecasting features."""
    n_samples = 25
    rng = np.random.default_rng(seed=42)

    data = pd.DataFrame(
        {
            "load": 100.0 + rng.normal(10.0, 5.0, n_samples),
            "temperature": 20.0 + rng.normal(1.0, 0.5, n_samples),
            "radiation": rng.uniform(0.0, 500.0, n_samples),
        },
        index=pd.date_range("2025-01-01 10:00", periods=n_samples, freq="h"),
    )

    return TimeSeriesDataset(data, timedelta(hours=1))


def test_forecasting_model__init__validates_horizons_and_assigns_components():
    """Test initialization validates horizon matching and assigns components correctly."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)

    preprocessing = FeatureEngineeringPipeline(horizons=horizons)
    postprocessing = TransformPipeline[ForecastDataset](transforms=[])

    # Act
    model = ForecastingModel(
        forecaster=forecaster,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        target_column="energy",
    )

    # Assert - Components are assigned correctly
    assert model.forecaster is forecaster
    assert model.preprocessing is preprocessing
    assert model.postprocessing is postprocessing
    assert model.target_column == "energy"
    assert model.is_fitted == forecaster.is_fitted  # Property delegation


def test_forecasting_model__init__uses_defaults():
    """Test initialization uses default preprocessing and postprocessing when not provided."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=1))]
    config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)

    # Act
    model = ForecastingModel(
        forecaster=forecaster, preprocessing=FeatureEngineeringPipeline(horizons=forecaster.config.horizons)
    )

    # Assert - Check that components are assigned correctly
    assert model.preprocessing is not None
    assert isinstance(model.preprocessing, FeatureEngineeringPipeline)
    assert model.preprocessing.horizons == forecaster.config.horizons
    assert model.postprocessing is not None
    assert isinstance(model.postprocessing, TransformPipeline)
    assert model.target_column == "load"  # Default value


def test_forecasting_model__init__raises_error_when_horizons_mismatch():
    """Test initialization raises error when forecaster and preprocessing horizons don't match."""
    # Arrange
    config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=[LeadTime(timedelta(hours=1))])
    forecaster = SimpleForecaster(config=config)
    preprocessing = MagicMock(spec=FeatureEngineeringPipeline)
    preprocessing.horizons = [LeadTime(timedelta(hours=6))]  # Different horizons

    # Act & Assert
    with pytest.raises(ConfigurationError, match=r"forecaster horizons.*do not match.*preprocessing horizons"):
        ForecastingModel(forecaster=forecaster, preprocessing=preprocessing)


@pytest.mark.parametrize(
    "forecaster_type",
    [
        pytest.param(HorizonForecaster, id="horizon_forecaster"),
        pytest.param(Forecaster, id="multi_horizon_forecaster"),
    ],
)
def test_forecasting_model__fit(forecaster_type: type, sample_timeseries_dataset: TimeSeriesDataset):
    """Test that fit correctly orchestrates preprocessing and forecaster calls, and returns metrics."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]

    if forecaster_type == HorizonForecaster:
        config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
        forecaster = SimpleForecaster(config=config)
    else:
        config = MultiHorizonForecasterConfig(
            quantiles=[Quantile(0.5)],
            horizons=horizons,
            forecaster_config=HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons),
        )
        forecaster = MultiHorizonForecasterAdapter.create(
            config=config, model_factory=lambda config: SimpleForecaster(config=config)
        )

    preprocessing = FeatureEngineeringPipeline(horizons=horizons)
    postprocessing = TransformPipeline[ForecastDataset](transforms=[])

    model = ForecastingModel(
        forecaster=forecaster,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
    )

    # Act
    result = model.fit(data=sample_timeseries_dataset)

    # Assert - Model is fitted and returns metrics
    assert model.is_fitted
    assert result.input_dataset is sample_timeseries_dataset
    assert result.input_data_train is not None
    assert result.metrics_train is not None
    assert result.metrics_full is not None
    assert result.metrics_val is not None
    assert result.metrics_test is not None
    # R2 metric exists for the 50th quantile
    assert Quantile(0.5) in result.metrics_train.metrics
    assert "R2" in result.metrics_train.metrics[Quantile(0.5)]


@pytest.mark.parametrize(
    "forecaster_type",
    [
        pytest.param(HorizonForecaster, id="horizon_forecaster"),
        pytest.param(Forecaster, id="multi_horizon_forecaster"),
    ],
)
def test_forecasting_model__predict(forecaster_type: type, sample_timeseries_dataset: TimeSeriesDataset):
    """Test that predict correctly orchestrates preprocessing and forecaster calls."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]

    if forecaster_type == HorizonForecaster:
        config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
        forecaster = SimpleForecaster(config=config)
    else:
        config = MultiHorizonForecasterConfig(
            quantiles=[Quantile(0.5)],
            horizons=horizons,
            forecaster_config=HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons),
        )
        forecaster = MultiHorizonForecasterAdapter.create(
            config=config, model_factory=lambda config: SimpleForecaster(config=config)
        )

    preprocessing = FeatureEngineeringPipeline(horizons=horizons)
    postprocessing = TransformPipeline[ForecastDataset](transforms=[])

    model = ForecastingModel(
        forecaster=forecaster,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
    )

    # Fit the model first
    model.fit(data=sample_timeseries_dataset)

    forecast_start = datetime.fromisoformat("2025-01-01T12:00:00")

    # Act
    result = model.predict(data=sample_timeseries_dataset, forecast_start=forecast_start)

    # Assert - Prediction returns a forecast dataset with expected properties
    assert isinstance(result, ForecastDataset)
    assert result.sample_interval == sample_timeseries_dataset.sample_interval
    assert result.quantiles == [Quantile(0.5)]
    assert result.forecast_start >= forecast_start
    assert not result.data.empty
    assert not result.data.isna().any().any()


def test_forecasting_model__predict__raises_error_when_not_fitted(sample_timeseries_dataset: TimeSeriesDataset):
    """Test predict raises NotFittedError when model is not fitted."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)
    preprocessing = FeatureEngineeringPipeline(horizons=horizons)

    model = ForecastingModel(forecaster=forecaster, preprocessing=preprocessing)

    # Act & Assert
    with pytest.raises(NotFittedError):
        model.predict(data=sample_timeseries_dataset)


def test_forecasting_model__score__returns_metrics(sample_timeseries_dataset: TimeSeriesDataset):
    """Test that score evaluates model and returns metrics."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)
    preprocessing = FeatureEngineeringPipeline(horizons=horizons)

    model = ForecastingModel(forecaster=forecaster, preprocessing=preprocessing)
    model.fit(data=sample_timeseries_dataset)

    # Act
    metrics = model.score(data=sample_timeseries_dataset)

    # Assert - Metrics are calculated for the median quantile
    assert metrics.metrics is not None
    assert Quantile(0.5) in metrics.metrics
    # R2 metric should be present (default evaluation metric)
    assert "R2" in metrics.metrics[Quantile(0.5)]


def test_forecasting_model__state_roundtrip(sample_timeseries_dataset: TimeSeriesDataset):
    """Test that model state can be serialized and restored."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    config = HorizonForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)
    preprocessing = FeatureEngineeringPipeline(horizons=horizons)

    original_model = ForecastingModel(forecaster=forecaster, preprocessing=preprocessing)
    original_model.fit(data=sample_timeseries_dataset)

    # Act - Serialize and restore
    state = original_model.to_state()
    restored_model = ForecastingModel(
        forecaster=SimpleForecaster(config=config), preprocessing=FeatureEngineeringPipeline(horizons=horizons)
    ).from_state(state)

    # Assert - Restored model is fitted and produces same predictions
    assert restored_model.is_fitted
    original_pred = original_model.predict(data=sample_timeseries_dataset)
    restored_pred = restored_model.predict(data=sample_timeseries_dataset)
    assert original_pred.data.equals(restored_pred.data)
