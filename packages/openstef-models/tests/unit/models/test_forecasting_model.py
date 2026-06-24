# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import PrivateAttr

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import InsufficientlyCompleteError, NotFittedError
from openstef_core.mixins import TransformPipeline
from openstef_core.mixins.param_ranges import IntRange
from openstef_core.mixins.predictor import HyperParams
from openstef_core.testing import assert_timeseries_equal, create_synthetic_forecasting_dataset
from openstef_core.types import LeadTime, Quantile, override
from openstef_models.models.forecasting.constant_quantile_forecaster import ConstantQuantileForecaster
from openstef_models.models.forecasting_model import ForecastingModel, restore_target
from openstef_models.testing import SimpleForecaster, create_sample_timeseries_dataset, create_standard_preprocessing
from openstef_models.transforms.general.outlier_handler import OUTLIER_NAN_MASK_PREFIX
from openstef_models.transforms.postprocessing.quantile_sorter import QuantileSorter


class TunableTestHyperParams(HyperParams):
    """Test hyperparameters that can carry a tuning override."""

    depth: int = 3


class TunableSimpleForecaster(SimpleForecaster):
    """Simple forecaster exposing unresolved tuning ranges for validation tests."""

    _hparams: TunableTestHyperParams = PrivateAttr(default_factory=TunableTestHyperParams)

    def __init__(self, *, quantiles: list[Quantile], horizons: list[LeadTime], supports_batching: bool = False) -> None:
        super().__init__(quantiles=quantiles, horizons=horizons, supports_batching=supports_batching)
        self._hparams = TunableTestHyperParams(depth=IntRange(1, 5, tune=True))  # ty: ignore[invalid-argument-type]

    @property
    @override
    def hparams(self) -> HyperParams:
        return self._hparams


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample time series data with typical energy forecasting features."""
    return create_sample_timeseries_dataset()


def test_forecasting_model__init__uses_defaults():
    """Test initialization uses default preprocessing and postprocessing when not provided."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=1))]
    forecaster = SimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons)

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

    forecaster = SimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons)
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

    forecaster = SimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons)
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
        ),
        sample_interval=timedelta(hours=1),
    )

    # Act & Assert
    with pytest.raises(InsufficientlyCompleteError):
        model.fit(data=data)

    assert not model.is_fitted


def test_forecasting_model__init_raises_for_unresolved_tuning_ranges() -> None:
    """Model construction fails fast when unresolved tuning ranges are still attached."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]

    # Act / Assert
    with pytest.raises(ValueError, match="unresolved tuning ranges"):
        ForecastingModel(forecaster=TunableSimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons))


def test_forecasting_model__predict(sample_timeseries_dataset: TimeSeriesDataset):
    """Test that predict correctly orchestrates preprocessing and forecaster calls."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]

    forecaster = SimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons)

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
    forecaster = SimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons)
    model = ForecastingModel(forecaster=forecaster)

    # Act & Assert
    with pytest.raises(NotFittedError):
        model.predict(data=sample_timeseries_dataset)


def test_forecasting_model__score__returns_metrics(sample_timeseries_dataset: TimeSeriesDataset):
    """Test that score evaluates model and returns metrics."""
    # Arrange
    horizons = [LeadTime(timedelta(hours=6))]
    forecaster = SimpleForecaster(quantiles=[Quantile(0.5)], horizons=horizons)

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
        forecaster=ConstantQuantileForecaster(
            quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
            horizons=horizons,
        ),
        preprocessing=create_standard_preprocessing(horizons),
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


def test_restore_target__preserves_outlier_nans_via_sentinel():
    """Verify that restore_target preserves NaN marked by OutlierHandler sentinel columns."""
    # Arrange
    index = pd.date_range("2025-01-01", periods=7, freq="1h")
    # Row 5 is originally NaN (missing data)
    original_data = pd.DataFrame({"load": [100.0, 200.0, 300.0, 400.0, 500.0, np.nan, 700.0]}, index=index)
    original_dataset = TimeSeriesDataset(original_data, timedelta(hours=1))

    preprocessed_data = pd.DataFrame(
        {
            "load": [100.0, np.nan, 300.0, np.nan, np.nan, np.nan, 700.0],
            f"{OUTLIER_NAN_MASK_PREFIX}load__": [False, True, False, True, False, False, False],
        },
        index=index,
    )
    preprocessed_dataset = TimeSeriesDataset(preprocessed_data, timedelta(hours=1))

    # Act
    result = restore_target(dataset=preprocessed_dataset, original_dataset=original_dataset, target_column="load")

    # Assert
    # - Rows 1, 3: remain NaN (outlier, sentinel=True)
    # - Row 4: restored to 500.0 (sentinel=False, so original value is used)
    # - Row 5: remains NaN (original is NaN)
    expected = pd.Series([100.0, np.nan, 300.0, np.nan, 500.0, np.nan, 700.0], index=index, name="load")
    pd.testing.assert_series_equal(result.data["load"], expected)
    # Sentinel column should be removed
    assert f"{OUTLIER_NAN_MASK_PREFIX}load__" not in result.data.columns


def test_restore_target__does_not_preserve_nan_without_sentinel():
    """Verify that restore_target restores target values when no sentinel column exists."""
    # Arrange
    index = pd.date_range("2025-01-01", periods=5, freq="1h")
    original_data = pd.DataFrame({"load": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=index)
    original_dataset = TimeSeriesDataset(original_data, timedelta(hours=1))

    # Preprocessed data with NaN but NO sentinel column (e.g., from some other source)
    preprocessed_data = pd.DataFrame({"load": [100.0, np.nan, 300.0, np.nan, 500.0]}, index=index)
    preprocessed_dataset = TimeSeriesDataset(preprocessed_data, timedelta(hours=1))

    # Act
    result = restore_target(dataset=preprocessed_dataset, original_dataset=original_dataset, target_column="load")

    # Assert - ALL values should be restored from original (NaN NOT preserved without sentinel)
    expected = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], index=index, name="load")
    pd.testing.assert_series_equal(result.data["load"], expected)


def test_restore_target__adds_target_when_not_present():
    """Verify that restore_target works when target column is missing (prediction output)."""
    # Arrange
    index = pd.date_range("2025-01-01", periods=3, freq="1h")
    original_data = pd.DataFrame({"load": [100.0, 200.0, 300.0]}, index=index)
    original_dataset = TimeSeriesDataset(original_data, timedelta(hours=1))

    # Prediction output without target column
    prediction_data = pd.DataFrame({"quantile_P50": [105.0, 195.0, 310.0]}, index=index)
    prediction_dataset = TimeSeriesDataset(prediction_data, timedelta(hours=1))

    # Act
    result = restore_target(dataset=prediction_dataset, original_dataset=original_dataset, target_column="load")

    # Assert - target column should be added with original values
    expected = pd.Series([100.0, 200.0, 300.0], index=index, name="load")
    pd.testing.assert_series_equal(result.data["load"], expected)
