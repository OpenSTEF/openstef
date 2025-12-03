# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pickle  # noqa: S403 - Controlled test
from datetime import datetime, timedelta
from typing import override

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastInputDataset
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins.predictor import HyperParams
from openstef_core.mixins.transform import TransformPipeline
from openstef_core.testing import assert_timeseries_equal, create_synthetic_forecasting_dataset
from openstef_core.types import LeadTime, Q
from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner, ForecastCombinerConfig
from openstef_meta.utils.datasets import EnsembleForecastDataset
from openstef_models.models.forecasting import Forecaster, ForecasterConfig
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


class SimpleCombiner(ForecastCombiner):
    """Simple combiner that averages base learner predictions."""

    def __init__(self, config: ForecastCombinerConfig):
        self._config = config
        self._is_fitted = False
        self.quantiles = config.quantiles

    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
        sample_weights: pd.Series | None = None,
    ) -> None:
        self._is_fitted = True

    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        if not self._is_fitted:
            raise NotFittedError("Combiner must be fitted before prediction.")

        combined_data = pd.DataFrame(index=data.data.index)
        for quantile in self.quantiles:
            quantile_cols = [col for col in data.data.columns if col.endswith(quantile.format())]
            combined_data[quantile.format()] = data.data[quantile_cols].mean(axis=1)

        return ForecastDataset(
            data=combined_data,
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


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


@pytest.fixture
def model() -> EnsembleForecastingModel:
    """Create a simple EnsembleForecastingModel for testing."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=1))]
    quantiles = [Q(0.3), Q(0.5), Q(0.7)]
    config = ForecasterConfig(quantiles=quantiles, horizons=horizons)
    forecasters: dict[str, Forecaster] = {
        "forecaster_1": SimpleForecaster(config=config),
        "forecaster_2": SimpleForecaster(config=config),
    }
    combiner_config = ForecastCombinerConfig(quantiles=quantiles, horizons=horizons, hyperparams=HyperParams())

    combiner = SimpleCombiner(
        config=combiner_config,
    )

    # Act
    return EnsembleForecastingModel(
        forecasters=forecasters, combiner=combiner, common_preprocessing=TransformPipeline()
    )


def test_forecasting_model__init__uses_defaults(model: EnsembleForecastingModel):
    """Test initialization uses default preprocessing and postprocessing when not provided."""

    # Assert - Check that components are assigned correctly
    assert model.common_preprocessing is not None
    assert model.postprocessing is not None
    assert model.target_column == "load"  # Default value
    assert model.forecaster_names == ["forecaster_1", "forecaster_2"]


def test_forecasting_model__fit(sample_timeseries_dataset: TimeSeriesDataset, model: EnsembleForecastingModel):
    """Test that fit correctly orchestrates preprocessing and forecaster calls, and returns metrics."""

    # Act
    result = model.fit(data=sample_timeseries_dataset)

    # Assert - Model is fitted and returns metrics
    assert model.is_fitted
    assert result is not None


def test_forecasting_model__predict(sample_timeseries_dataset: TimeSeriesDataset, model: EnsembleForecastingModel):
    """Test that predict correctly orchestrates preprocessing and forecaster calls."""

    # Fit the model first
    model.fit(data=sample_timeseries_dataset)
    forecast_start = datetime.fromisoformat("2025-01-01T12:00:00")

    # Act
    result = model.predict(data=sample_timeseries_dataset, forecast_start=forecast_start)

    # Assert - Prediction returns a forecast dataset with expected properties
    assert isinstance(result, ForecastDataset)
    assert result.sample_interval == sample_timeseries_dataset.sample_interval
    assert result.quantiles == [Q(0.3), Q(0.5), Q(0.7)]
    assert result.forecast_start >= forecast_start
    assert not result.data.empty
    assert not result.data.isna().any().any()


def test_forecasting_model__predict__raises_error_when_not_fitted(
    sample_timeseries_dataset: TimeSeriesDataset, model: EnsembleForecastingModel
):
    """Test predict raises NotFittedError when model is not fitted."""

    # Act & Assert
    with pytest.raises(NotFittedError):
        model.predict(data=sample_timeseries_dataset)


def test_forecasting_model__score__returns_metrics(
    sample_timeseries_dataset: TimeSeriesDataset, model: EnsembleForecastingModel
):
    """Test that score evaluates model and returns metrics."""

    model.fit(data=sample_timeseries_dataset)

    # Act
    metrics = model.score(data=sample_timeseries_dataset)

    # Assert - Metrics are calculated for the median quantile
    assert metrics.metrics is not None
    assert all(x in metrics.metrics for x in [Q(0.3), Q(0.5), Q(0.7)])
    # R2 metric should be present (default evaluation metric)
    assert "R2" in metrics.metrics[Q(0.5)]


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
    # Arrange
    horizons = [LeadTime(timedelta(hours=1))]
    quantiles = [Q(0.3), Q(0.5), Q(0.7)]
    config = ForecasterConfig(quantiles=quantiles, horizons=horizons)
    forecasters: dict[str, Forecaster] = {
        "forecaster_1": SimpleForecaster(config=config),
        "forecaster_2": SimpleForecaster(config=config),
    }
    combiner_config = ForecastCombinerConfig(quantiles=quantiles, horizons=horizons, hyperparams=HyperParams())

    combiner = SimpleCombiner(
        config=combiner_config,
    )

    original_model = EnsembleForecastingModel(
        forecasters=forecasters,
        combiner=combiner,
        common_preprocessing=TransformPipeline(
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
    assert isinstance(restored_model, EnsembleForecastingModel)
    assert restored_model.is_fitted
    assert restored_model.target_column == original_model.target_column
    assert restored_model.cutoff_history == original_model.cutoff_history

    # Verify predictions match using pandas testing utilities
    actual_predictions = restored_model.predict(data=dataset)
    assert_timeseries_equal(actual_predictions, expected_predictions)
