# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for validated dataset classes."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import (
    EnergyComponentDataset,
    EnsembleForecastDataset,
    ForecastDataset,
    ForecastInputDataset,
)
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import LeadTime, Quantile


@pytest.fixture
def forecast_index() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01T00:00:00", periods=6, freq="h")


# ---------------------------------------------------------------------------
# ForecastInputDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_input_dataset(forecast_index: pd.DatetimeIndex) -> ForecastInputDataset:
    data = pd.DataFrame(
        {
            "load": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "temperature": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "wind_speed": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        },
        index=forecast_index,
    )
    return ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
    )


@pytest.fixture
def weighted_input_dataset(forecast_index: pd.DatetimeIndex) -> ForecastInputDataset:
    data = pd.DataFrame(
        {
            "load": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "temperature": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "sample_weight": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
        },
        index=forecast_index,
    )
    return ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        sample_weight_column="sample_weight",
    )


def test_forecast_input_dataset__missing_target_column_raises():
    """Missing target column raises MissingColumnsError at construction."""
    # Arrange
    data = pd.DataFrame({"temperature": [10.0]}, index=pd.date_range("2025-01-01", periods=1, freq="h"))

    # Act & Assert
    with pytest.raises(MissingColumnsError):
        ForecastInputDataset(data=data, sample_interval=timedelta(hours=1), target_column="load")


def test_sample_weight_series__returns_ones_when_column_missing(basic_input_dataset: ForecastInputDataset):
    """sample_weight_series returns all-ones Series when weight column is absent."""
    # Act
    weights = basic_input_dataset.sample_weight_series

    # Assert
    assert (weights == 1).all()
    assert len(weights) == len(basic_input_dataset.index)


def test_sample_weight_series__returns_column_values(weighted_input_dataset: ForecastInputDataset):
    """sample_weight_series returns the actual weight column values."""
    # Arrange
    expected = pd.Series([1.0, 0.5, 1.0, 0.5, 1.0, 0.5], index=weighted_input_dataset.index, name="sample_weight")

    # Act
    weights = weighted_input_dataset.sample_weight_series

    # Assert
    pd.testing.assert_series_equal(weights, expected)


def test_input_data__expected_columns(basic_input_dataset: ForecastInputDataset):
    """input_data includes expected columns."""
    # Act
    result = basic_input_dataset.input_data()

    # Assert
    assert "load" not in result.columns
    assert "sample_weight" not in result.columns
    assert "temperature" in result.columns
    assert "wind_speed" in result.columns


def test_input_data__start_filters_rows(basic_input_dataset: ForecastInputDataset):
    """Rows before start are excluded when start is provided."""
    # Arrange
    start = datetime(2025, 1, 1, 3, 0)

    # Act
    result = basic_input_dataset.input_data(start=start)

    # Assert
    assert result.index.min() >= pd.Timestamp(start)
    assert len(result) == 3


def test_input_data__no_start_returns_all_rows(basic_input_dataset: ForecastInputDataset):
    """All rows returned when start is None."""
    # Act
    result = basic_input_dataset.input_data()

    # Assert
    assert len(result) == 6


@pytest.mark.parametrize(
    ("horizon_hours", "expected_len"),
    [
        pytest.param(2, 3, id="2h_horizon"),
        pytest.param(5, 6, id="5h_horizon_full_range"),
        pytest.param(0, 1, id="0h_horizon_only_start"),
    ],
)
def test_input_data__horizon_limits_rows(
    basic_input_dataset: ForecastInputDataset,
    horizon_hours: int,
    expected_len: int,
):
    """Rows after forecast_start + horizon are excluded when horizon is given."""
    # Arrange
    horizon = LeadTime(timedelta(hours=horizon_hours))

    # Act
    # forecast_start = 2025-01-01T00:00 (first index), so end = 00:00 + horizon
    result = basic_input_dataset.input_data(start=basic_input_dataset.forecast_start, horizon=horizon)

    # Assert
    assert len(result) == expected_len


def test_input_data__horizon_none_returns_all_rows(basic_input_dataset: ForecastInputDataset):
    """No filtering by horizon when horizon=None."""
    # Act
    result = basic_input_dataset.input_data(start=basic_input_dataset.forecast_start, horizon=None)

    # Assert
    assert len(result) == 6


def test_input_data__horizon_combined_with_start(basic_input_dataset: ForecastInputDataset):
    """start and horizon together correctly bound the returned rows."""
    # Arrange
    # forecast_start = 2025-01-01T00:00, start = T+2h, horizon = 4h → end = T+4h
    # expected rows: T+2h, T+3h, T+4h → 3 rows
    start = datetime(2025, 1, 1, 2, 0)
    horizon = LeadTime(timedelta(hours=4))

    # Act
    result = basic_input_dataset.input_data(start=start, horizon=horizon)

    # Assert
    assert len(result) == 3
    assert result.index.min() == pd.Timestamp(start)
    assert result.index.max() == pd.Timestamp(basic_input_dataset.forecast_start + timedelta(hours=4))


def test_create_forecast_range__correct_range(basic_input_dataset: ForecastInputDataset):
    """Forecast range ends at forecast_start + horizon."""
    # Arrange
    horizon = LeadTime(timedelta(hours=3))
    expected_end = pd.Timestamp(basic_input_dataset.forecast_start + timedelta(hours=3))

    # Act
    index = basic_input_dataset.create_forecast_range(horizon=horizon)

    # Assert
    assert len(index) == 4  # 00:00, 01:00, 02:00, 03:00
    assert index[0] == pd.Timestamp(basic_input_dataset.forecast_start)
    assert index[-1] == expected_end


def test_to_pandas_roundtrip(basic_input_dataset: ForecastInputDataset):
    """to_pandas / reconstruction preserves target_column and forecast_start in attrs."""
    # Act
    df = basic_input_dataset.to_pandas()

    # Assert
    assert df.attrs["target_column"] == "load"
    assert "forecast_start" in df.attrs

    restored = ForecastInputDataset(data=df, sample_interval=timedelta(hours=1))
    assert restored.target_column == basic_input_dataset.target_column
    assert restored.forecast_start == basic_input_dataset.forecast_start


# ---------------------------------------------------------------------------
# ForecastDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def forecast_dataset(forecast_index: pd.DatetimeIndex) -> ForecastDataset:
    data = pd.DataFrame(
        {
            "quantile_P10": [90.0, 100.0, 110.0, 120.0, 130.0, 140.0],
            "quantile_P50": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "quantile_P90": [110.0, 120.0, 130.0, 140.0, 150.0, 160.0],
        },
        index=forecast_index,
    )
    return ForecastDataset(data=data, sample_interval=timedelta(hours=1))


def test_forecast_dataset__invalid_column_raises():
    """ForecastDataset raises when a non-quantile column name is present."""
    # Arrange
    data = pd.DataFrame(
        {"not_a_quantile": [1.0, 2.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )

    # Act & Assert
    with pytest.raises(ValueError, match="valid quantile"):
        ForecastDataset(data=data, sample_interval=timedelta(hours=1))


def test_forecast_dataset__median_series(forecast_dataset: ForecastDataset):
    """median_series returns the P50 column."""
    # Arrange
    expected = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0], index=forecast_dataset.index, name="quantile_P50")

    # Act
    median = forecast_dataset.median_series

    # Assert
    pd.testing.assert_series_equal(median, expected)


def test_forecast_dataset__median_series_missing_raises(forecast_index: pd.DatetimeIndex):
    """median_series raises MissingColumnsError when P50 is absent."""
    # Arrange
    data = pd.DataFrame({"quantile_P10": [1.0]}, index=pd.date_range("2025-01-01", periods=1, freq="h"))
    ds = ForecastDataset(data=data, sample_interval=timedelta(hours=1))

    # Act & Assert
    with pytest.raises(MissingColumnsError):
        ds.median_series


def test_forecast_dataset__target_series_none_when_absent(forecast_dataset: ForecastDataset):
    """target_series is None when target column is not in the data."""
    # Act & Assert
    assert forecast_dataset.target_series is None


def test_forecast_dataset__target_series_returned_when_present(forecast_index: pd.DatetimeIndex):
    """target_series returns the column when it exists."""
    # Arrange
    data = pd.DataFrame(
        {
            "quantile_P50": [100.0, 110.0],
            "load": [95.0, 105.0],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    ds = ForecastDataset(data=data, sample_interval=timedelta(hours=1), target_column="load")

    # Act & Assert
    assert ds.target_series is not None
    assert list(ds.target_series) == [95.0, 105.0]


def test_forecast_dataset__filter_quantiles(forecast_dataset: ForecastDataset):
    """filter_quantiles keeps only the requested quantile columns."""
    # Act
    result = forecast_dataset.filter_quantiles([Quantile(0.1), Quantile(0.9)])

    # Assert
    assert [float(q) for q in result.quantiles] == pytest.approx([0.1, 0.9], abs=1e-9)
    assert "quantile_P50" not in result.data.columns


def test_forecast_dataset__filter_quantiles_missing_raises(forecast_dataset: ForecastDataset):
    """filter_quantiles raises MissingColumnsError for a non-existent quantile."""
    # Act & Assert
    with pytest.raises(MissingColumnsError):
        forecast_dataset.filter_quantiles([Quantile(0.05)])


def test_forecast_dataset__from_quantile_predictions():
    """from_quantile_predictions builds a ForecastDataset from a raw numpy array."""
    # Arrange
    quantiles = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]
    predictions = np.array([[90.0, 100.0, 110.0], [95.0, 105.0, 115.0]])
    index = pd.date_range("2025-01-01", periods=2, freq="h")

    # Act
    ds = ForecastDataset.from_quantile_predictions(predictions, index, quantiles, timedelta(hours=1))

    # Assert
    assert sorted(float(q) for q in ds.quantiles) == [0.1, 0.5, 0.9]
    assert len(ds.data) == 2


# ---------------------------------------------------------------------------
# EnergyComponentDataset
# ---------------------------------------------------------------------------


def test_energy_component_dataset__valid():
    """EnergyComponentDataset accepts data with all required energy component columns."""
    # Arrange
    data = pd.DataFrame(
        {"wind": [50.0], "solar": [30.0], "other": [20.0]},
        index=pd.date_range("2025-01-01", periods=1, freq="h"),
    )

    # Act
    ds = EnergyComponentDataset(data=data, sample_interval=timedelta(hours=1))

    # Assert
    assert set(ds.feature_names) == {"wind", "solar", "other"}


def test_energy_component_dataset__missing_component_raises():
    """EnergyComponentDataset raises when a required energy component column is absent."""
    # Arrange
    data = pd.DataFrame(
        {"wind": [50.0], "solar": [30.0]},
        index=pd.date_range("2025-01-01", periods=1, freq="h"),
    )

    # Act & Assert
    with pytest.raises(MissingColumnsError):
        EnergyComponentDataset(data=data, sample_interval=timedelta(hours=1))


# ---------------------------------------------------------------------------
# EnsembleForecastDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def ensemble_dataset(forecast_index: pd.DatetimeIndex) -> EnsembleForecastDataset:
    data = pd.DataFrame(
        {
            "lgbm__quantile_P10": np.full(6, 90.0),
            "lgbm__quantile_P50": np.full(6, 100.0),
            "xgb__quantile_P10": np.full(6, 88.0),
            "xgb__quantile_P50": np.full(6, 99.0),
            "load": np.linspace(95.0, 105.0, 6),
        },
        index=forecast_index,
    )
    return EnsembleForecastDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
    )


def test_ensemble_forecast_dataset__forecaster_names_parsed(ensemble_dataset: EnsembleForecastDataset):
    """EnsembleForecastDataset parses forecaster names from column prefixes."""
    # Act & Assert
    assert set(ensemble_dataset.forecaster_names) == {"lgbm", "xgb"}


def test_ensemble_forecast_dataset__quantiles_parsed(ensemble_dataset: EnsembleForecastDataset):
    """EnsembleForecastDataset parses quantiles from column suffixes."""
    # Act & Assert
    assert sorted(float(q) for q in ensemble_dataset.quantiles) == [0.1, 0.5]


def test_ensemble_forecast_dataset__get_base_predictions_for_quantile(ensemble_dataset: EnsembleForecastDataset):
    """get_base_predictions_for_quantile returns ForecastInputDataset with correct columns."""
    # Act
    result = ensemble_dataset.get_base_predictions_for_quantile(Quantile(0.1))

    # Assert
    assert isinstance(result, ForecastInputDataset)
    # feature_names includes all non-internal columns; the forecaster columns plus target_column
    assert {"lgbm", "xgb"}.issubset(set(result.data.columns))


def test_ensemble_forecast_dataset__from_forecast_datasets(forecast_index: pd.DatetimeIndex):
    """from_forecast_datasets correctly merges multiple ForecastDatasets."""
    # Arrange
    ds_a = ForecastDataset(
        data=pd.DataFrame({"quantile_P50": [100.0] * 3}, index=forecast_index[:3]),
        sample_interval=timedelta(hours=1),
    )
    ds_b = ForecastDataset(
        data=pd.DataFrame({"quantile_P50": [110.0] * 3}, index=forecast_index[:3]),
        sample_interval=timedelta(hours=1),
    )

    # Act
    ensemble = EnsembleForecastDataset.from_forecast_datasets({"a": ds_a, "b": ds_b})

    # Assert
    assert set(ensemble.forecaster_names) == {"a", "b"}
    assert len(ensemble.data) == 3


def test_ensemble_forecast_dataset__column_missing_separator_raises(forecast_index: pd.DatetimeIndex):
    """EnsembleForecastDataset raises ValueError for columns lacking the separator."""
    # Arrange
    data = pd.DataFrame(
        {"noseperator_quantile_P50": [1.0]},
        index=pd.date_range("2025-01-01", periods=1, freq="h"),
    )

    # Act & Assert
    with pytest.raises(ValueError):
        EnsembleForecastDataset(data=data, sample_interval=timedelta(hours=1))
