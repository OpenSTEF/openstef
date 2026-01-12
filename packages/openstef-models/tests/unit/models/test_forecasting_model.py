# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pickle  # noqa: S403 - controlled test
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import InsufficientlyCompleteError, NotFittedError
from openstef_core.mixins import TransformPipeline
from openstef_core.testing import assert_timeseries_equal, create_synthetic_forecasting_dataset
from openstef_core.types import LeadTime, Quantile, override
from openstef_models.models.forecasting.constant_median_forecaster import (
    ConstantMedianForecaster,
    ConstantMedianForecasterConfig,
)
from openstef_models.models.forecasting.forecaster import Forecaster, ForecasterConfig
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.transforms.postprocessing.quantile_sorter import QuantileSorter
from openstef_models.transforms.time_domain.lags_adder import LagsAdder


class SimpleForecaster(Forecaster):
    """Simple test forecaster that returns predictable values for testing."""

    def __init__(self, config: ForecasterConfig):
        self._config = config
        self._is_fitted = False

    @property
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

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


def test_forecasting_model__init__uses_defaults():
    """Test initialization uses default preprocessing and postprocessing when not provided."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=1))]
    config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)

    # Act
    model = ForecastingModel(forecaster=forecaster, preprocessing=TransformPipeline())

    # Assert - Check that components are assigned correctly
    assert model.preprocessing is not None
    assert model.postprocessing is not None
    assert model.target_column == "load"  # Default value


def test_forecasting_model__fit(sample_timeseries_dataset: TimeSeriesDataset):
    """Test that fit correctly orchestrates preprocessing and forecaster calls, and returns metrics."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]

    config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)
    model = ForecastingModel(forecaster=forecaster)

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


def test_forecasting_model__fit_all_nan_target():
    """Test that fit raises error when fitted on all-NaN target column."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]

    config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)
    model = ForecastingModel(forecaster=forecaster)
    n_samples = 25
    data = TimeSeriesDataset(
        data=pd.DataFrame(
        {
            "load": [np.nan] * n_samples,
            "temperature": [20.0] * n_samples,
            "radiation": [500.0] * n_samples,
        },
        index=pd.date_range("2025-01-01 10:00", periods=n_samples, freq="h"),
    ), sample_interval=timedelta(hours=1)
    )

    # Act & Assert
    with pytest.raises(InsufficientlyCompleteError):
        model.fit(data=data)

    assert not model.is_fitted


def test_forecasting_model__predict(sample_timeseries_dataset: TimeSeriesDataset):
    """Test that predict correctly orchestrates preprocessing and forecaster calls."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]

    config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)

    model = ForecastingModel(forecaster=forecaster)

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
    config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)
    model = ForecastingModel(forecaster=forecaster)

    # Act & Assert
    with pytest.raises(NotFittedError):
        model.predict(data=sample_timeseries_dataset)


def test_forecasting_model__score__returns_metrics(sample_timeseries_dataset: TimeSeriesDataset):
    """Test that score evaluates model and returns metrics."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    config = ForecasterConfig(quantiles=[Quantile(0.5)], horizons=horizons)
    forecaster = SimpleForecaster(config=config)

    model = ForecastingModel(forecaster=forecaster)
    model.fit(data=sample_timeseries_dataset)

    # Act
    metrics = model.score(data=sample_timeseries_dataset)

    # Assert - Metrics are calculated for the median quantile
    assert metrics.metrics is not None
    assert Quantile(0.5) in metrics.metrics
    # R2 metric should be present (default evaluation metric)
    assert "R2" in metrics.metrics[Quantile(0.5)]


def test_forecasting_model__pickle_roundtrip():
    """Test that ForecastingModel with preprocessing and postprocessing can be pickled and unpickled.

    This verifies that the entire forecasting pipeline, including transforms and forecaster,
    can be serialized and deserialized while maintaining functionality.
    """
    # Arrange - create synthetic dataset
    dataset = create_synthetic_forecasting_dataset(
        length=timedelta(days=30),
        sample_interval=timedelta(hours=1),
        random_seed=42,
    )

    # Create forecasting model with preprocessing and postprocessing
    horizons = [LeadTime(timedelta(hours=6))]

    original_model = ForecastingModel(
        forecaster=ConstantMedianForecaster(
            config=ConstantMedianForecasterConfig(
                quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
                horizons=horizons,
            )
        ),
        preprocessing=TransformPipeline(
            transforms=[
                LagsAdder(
                    history_available=timedelta(days=14),
                    horizons=horizons,
                    max_day_lags=7,
                    add_trivial_lags=True,
                    add_autocorr_lags=False,
                ),
            ]
        ),
        postprocessing=TransformPipeline(transforms=[QuantileSorter()]),
        cutoff_history=timedelta(days=7),
        target_column="load",
    )

    # Fit the original model
    original_model.fit(data=dataset)

    # Get predictions from original model
    expected_predictions = original_model.predict(data=dataset)

    # Act - pickle and unpickle the model
    pickled = pickle.dumps(original_model)
    restored_model = pickle.loads(pickled)  # noqa: S301 - Controlled test

    # Assert - verify the restored model is the correct type
    assert isinstance(restored_model, ForecastingModel)
    assert restored_model.is_fitted
    assert restored_model.target_column == original_model.target_column
    assert restored_model.cutoff_history == original_model.cutoff_history

    # Verify predictions match using pandas testing utilities
    actual_predictions = restored_model.predict(data=dataset)
    assert_timeseries_equal(actual_predictions, expected_predictions)
