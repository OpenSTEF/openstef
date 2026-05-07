# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pickle  # noqa: S403 - Controlled test
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, override

import pandas as pd
import pytest

from openstef_core.datasets import ForecastInputDataset
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import EnsembleForecastDataset, ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins.predictor import HyperParams
from openstef_core.mixins.transform import TransformPipeline
from openstef_core.testing import assert_timeseries_equal, create_synthetic_forecasting_dataset
from openstef_core.types import LeadTime, Q
from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner
from openstef_models.testing import SimpleForecaster, create_sample_timeseries_dataset, create_standard_preprocessing
from openstef_models.transforms.postprocessing.quantile_sorter import QuantileSorter

if TYPE_CHECKING:
    from openstef_models.models.forecasting.forecaster import Forecaster


class SimpleCombiner(ForecastCombiner):
    """Simple combiner that averages base Forecaster predictions."""

    @property
    @override
    def hparams(self) -> HyperParams:
        return HyperParams()

    _is_fitted: bool = False

    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
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

    @property
    def feature_importances(self) -> pd.DataFrame:
        return pd.DataFrame()

    @override
    def predict_contributions(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> TimeSeriesDataset:
        return TimeSeriesDataset(pd.DataFrame(index=data.data.index), data.sample_interval)


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample time series data with typical energy forecasting features."""
    return create_sample_timeseries_dataset()


@pytest.fixture
def model() -> EnsembleForecastingModel:
    """Create a simple EnsembleForecastingModel for testing."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=1))]
    quantiles = [Q(0.3), Q(0.5), Q(0.7)]
    forecasters: dict[str, Forecaster] = {
        "forecaster_1": SimpleForecaster(quantiles=quantiles, horizons=horizons),
        "forecaster_2": SimpleForecaster(quantiles=quantiles, horizons=horizons),
    }

    combiner = SimpleCombiner(quantiles=quantiles, horizons=horizons)

    # Act
    return EnsembleForecastingModel(forecasters=forecasters, combiner=combiner, preprocessing=TransformPipeline())


def test_forecasting_model__fit(sample_timeseries_dataset: TimeSeriesDataset, model: EnsembleForecastingModel):
    """Test that fit correctly orchestrates preprocessing and forecaster calls, and returns metrics."""
    # Act
    result = model.fit(data=sample_timeseries_dataset)

    # Assert
    assert model.is_fitted
    assert result is not None


def test_forecasting_model__predict(sample_timeseries_dataset: TimeSeriesDataset, model: EnsembleForecastingModel):
    """Test that predict correctly orchestrates preprocessing and forecaster calls."""
    # Arrange
    model.fit(data=sample_timeseries_dataset)
    forecast_start = datetime.fromisoformat("2025-01-01T12:00:00")

    # Act
    result = model.predict(data=sample_timeseries_dataset, forecast_start=forecast_start)

    # Assert
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
    # Arrange
    model.fit(data=sample_timeseries_dataset)

    # Act
    metrics = model.score(data=sample_timeseries_dataset)

    # Assert
    assert metrics.metrics is not None
    assert all(x in metrics.metrics for x in [Q(0.3), Q(0.5), Q(0.7)])
    assert "R2" in metrics.metrics[Q(0.5)]


def test_forecasting_model__pickle_roundtrip():
    """Test that ForecastingModel with preprocessing and postprocessing can be pickled and unpickled.

    This verifies that the entire forecasting pipeline, including transforms and forecaster,
    can be serialized and deserialized while maintaining functionality.
    """
    # Arrange
    dataset = create_synthetic_forecasting_dataset(
        length=timedelta(days=30),
        sample_interval=timedelta(hours=1),
        random_seed=42,
    )

    horizons = [LeadTime(timedelta(hours=1))]
    quantiles = [Q(0.3), Q(0.5), Q(0.7)]
    forecasters: dict[str, Forecaster] = {
        "forecaster_1": SimpleForecaster(quantiles=quantiles, horizons=horizons),
        "forecaster_2": SimpleForecaster(quantiles=quantiles, horizons=horizons),
    }

    combiner = SimpleCombiner(quantiles=quantiles, horizons=horizons)

    original_model = EnsembleForecastingModel(
        forecasters=forecasters,
        combiner=combiner,
        preprocessing=create_standard_preprocessing(horizons),
        postprocessing=TransformPipeline(transforms=[QuantileSorter()]),
        cutoff_history=timedelta(days=7),
        target_column="load",
    )

    # Fit the original model
    original_model.fit(data=dataset)

    # Get predictions from original model
    expected_predictions = original_model.predict(data=dataset)

    # Act
    pickled = pickle.dumps(original_model)
    restored_model = pickle.loads(pickled)  # noqa: S301 - Controlled test

    # Assert
    assert isinstance(restored_model, EnsembleForecastingModel)
    assert restored_model.is_fitted
    assert restored_model.target_column == original_model.target_column
    assert restored_model.cutoff_history == original_model.cutoff_history

    # Verify predictions match using pandas testing utilities
    actual_predictions = restored_model.predict(data=dataset)
    assert_timeseries_equal(actual_predictions, expected_predictions)
