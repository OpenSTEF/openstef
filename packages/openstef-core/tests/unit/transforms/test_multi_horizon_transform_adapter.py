# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from typing import Any, Self, cast, override

import pandas as pd
import pytest

from openstef_core.datasets import MultiHorizon, TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.mixins import State
from openstef_core.transforms.dataset_transforms import TimeSeriesTransform
from openstef_core.transforms.horizon_split_transform import (
    HorizonSplitTransform,
)
from openstef_core.transforms.multi_horizon_transform_adapter import (
    MultiHorizonTransformAdapter,
    concat_horizon_datasets_rowwise,
)
from openstef_core.types import LeadTime


class DummyTransform(TimeSeriesTransform):
    """A simple transform for testing that adds a constant to numeric columns."""

    def __init__(self, constant: float = 1.0) -> None:
        self.constant = constant
        self._fitted = False

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        self._fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        transformed_data = data.data.copy()
        numeric_columns = transformed_data.select_dtypes(include=[float, int]).columns
        transformed_data[numeric_columns] += self.constant
        return TimeSeriesDataset(transformed_data, data.sample_interval)

    @override
    def features_added(self) -> list[str]:
        return []

    def to_state(self) -> State:
        return {"constant": self.constant, "fitted": self._fitted}

    def from_state(self, state: dict[str, Any]) -> Self:  # type: ignore[override]
        self.constant = state["constant"]
        self._fitted = state["fitted"]
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted


@pytest.fixture
def sample_multi_horizon_data() -> MultiHorizon[TimeSeriesDataset]:
    """Create sample multi-horizon dataset for testing."""
    data1 = pd.DataFrame(
        {"load": [100.0, 110.0], "temperature": [20.0, 21.0]},
        index=pd.date_range("2025-01-01 10:00", periods=2, freq="1h"),
    )
    dataset1 = TimeSeriesDataset(data1, timedelta(hours=1))

    data2 = pd.DataFrame(
        {"load": [120.0, 130.0], "temperature": [22.0, 23.0]},
        index=pd.date_range("2025-01-01 12:00", periods=2, freq="1h"),
    )
    dataset2 = TimeSeriesDataset(data2, timedelta(hours=1))

    return MultiHorizon({
        LeadTime.from_string("PT1H"): dataset1,
        LeadTime.from_string("PT2H"): dataset2,
    })


@pytest.fixture
def dummy_transform() -> DummyTransform:
    """Create a dummy transform instance."""
    return DummyTransform(constant=5.0)


@pytest.fixture
def adapter(dummy_transform: DummyTransform) -> MultiHorizonTransformAdapter:
    """Create an adapter with dummy transform."""
    return MultiHorizonTransformAdapter(time_series_transform=dummy_transform)


@pytest.mark.parametrize(
    ("datasets_config", "expected_behavior"),
    [
        pytest.param(
            [{"load": [100.0, 110.0], "temperature": [20.0, 21.0], "start": "2025-01-01 10:00", "periods": 2}],
            {"type": "single", "sample_interval": timedelta(hours=1)},
            id="single_dataset",
        ),
        pytest.param(
            [
                {"load": [100.0, 110.0], "temperature": [20.0, 21.0], "start": "2025-01-01 10:00", "periods": 2},
                {"load": [120.0, 130.0], "temperature": [22.0, 23.0], "start": "2025-01-01 12:00", "periods": 2},
            ],
            {"type": "multiple", "sample_interval": timedelta(hours=1), "total_rows": 4},
            id="multiple_datasets",
        ),
        pytest.param(
            [
                {"load": [100.0, 110.0], "temperature": [20.0, 21.0], "start": "2025-01-01 10:00", "periods": 2},
                {"load": [105.0, 115.0], "temperature": [20.5, 21.5], "start": "2025-01-01 10:00", "periods": 2},
            ],
            {"type": "overlapping", "sample_interval": timedelta(hours=1), "total_rows": 4},
            id="handles_overlapping_indices",
        ),
    ],
)
def test_concat_horizon_datasets_rowwise(datasets_config: list[dict[str, Any]], expected_behavior: dict[str, Any]):
    """Test concatenating horizon datasets with various configurations."""
    # Arrange
    horizon_datasets_dict: dict[LeadTime, TimeSeriesDataset] = {}
    expected_sample_interval = expected_behavior["sample_interval"]

    for i, config in enumerate(datasets_config):
        data_dict = {k: v for k, v in config.items() if k not in {"start", "periods", "interval"}}
        index = pd.date_range(config["start"], periods=config["periods"], freq="1h")
        data = pd.DataFrame(data_dict, index=index)

        interval = config.get("interval", timedelta(hours=1))
        dataset = TimeSeriesDataset(data, sample_interval=interval)
        horizon_datasets_dict[LeadTime.from_string(f"PT{i + 1}H")] = dataset

    horizon_datasets = MultiHorizon(horizon_datasets_dict)

    # Act
    result = concat_horizon_datasets_rowwise(horizon_datasets)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.sample_interval == expected_sample_interval

    if expected_behavior["type"] == "single":
        # Single dataset should be identical to input
        original_dataset = next(iter(horizon_datasets_dict.values()))
        pd.testing.assert_frame_equal(result.data, original_dataset.data)
    elif "total_rows" in expected_behavior:
        # Check expected number of rows
        assert len(result.data) == expected_behavior["total_rows"]


def test_concat_horizon_datasets_rowwise_empty_datasets():
    """Test concatenation with empty datasets."""
    # Arrange
    empty_data = pd.DataFrame(columns=["load", "temperature"])
    empty_data.index = pd.DatetimeIndex([], name="timestamp")
    empty_dataset = TimeSeriesDataset(empty_data, timedelta(hours=1))
    horizon_datasets = MultiHorizon({LeadTime.from_string("PT1H"): empty_dataset})

    # Act
    result = concat_horizon_datasets_rowwise(horizon_datasets)

    # Assert
    assert isinstance(result, TimeSeriesDataset)
    assert result.data.empty
    assert result.sample_interval == empty_dataset.sample_interval


def test_horizon_split_transform_integration_with_concat():
    """Test that HorizonSplitTransform output can be properly concatenated."""
    # Arrange
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01T10:00:00",
            "2025-01-01T11:00:00",
        ]),
        "available_at": pd.to_datetime([
            "2025-01-01T10:05:00",
            "2025-01-01T11:05:00",
        ]),
        "load": [100.0, 110.0],
    })
    versioned_dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))

    horizons = [LeadTime.from_string("PT1H"), LeadTime.from_string("PT2H")]
    transform = HorizonSplitTransform(horizons=horizons)

    # Act
    horizon_datasets = transform.transform(versioned_dataset)
    concatenated = concat_horizon_datasets_rowwise(horizon_datasets)

    # Assert
    assert isinstance(concatenated, TimeSeriesDataset)
    assert "load" in concatenated.data.columns
    assert concatenated.sample_interval == versioned_dataset.sample_interval


def test_multi_horizon_transform_adapter_initialization(dummy_transform: DummyTransform):
    """Test that MultiHorizonTransformAdapter initializes correctly."""
    # Arrange & Act
    adapter = MultiHorizonTransformAdapter(time_series_transform=dummy_transform)

    # Assert
    # Adapter should be created successfully
    assert adapter.time_series_transform is dummy_transform
    assert not adapter.is_fitted


def test_multi_horizon_transform_adapter_is_fitted_delegates_to_underlying(
    adapter: MultiHorizonTransformAdapter, dummy_transform: DummyTransform
):
    """Test that is_fitted property delegates to the underlying transform."""
    # Arrange
    assert not adapter.is_fitted

    # Act
    dummy_transform._fitted = True

    # Assert
    # is_fitted should reflect the underlying transform's state
    assert adapter.is_fitted


def test_multi_horizon_transform_adapter_fit_sets_fitted_state(
    adapter: MultiHorizonTransformAdapter, sample_multi_horizon_data: MultiHorizon[TimeSeriesDataset]
):
    """Test that fit method sets the fitted state correctly."""
    # Arrange
    assert not adapter.is_fitted

    # Act
    adapter.fit(sample_multi_horizon_data)

    # Assert
    # Adapter should be fitted after calling fit
    assert adapter.is_fitted


def test_multi_horizon_transform_adapter_transform_applies_to_each_horizon(
    adapter: MultiHorizonTransformAdapter, sample_multi_horizon_data: MultiHorizon[TimeSeriesDataset]
):
    """Test that transform applies the underlying transform to each horizon independently."""
    # Arrange
    adapter.fit(sample_multi_horizon_data)
    constant = 5.0  # From dummy_transform fixture

    # Act
    result: MultiHorizon[TimeSeriesDataset] = adapter.transform(sample_multi_horizon_data)

    # Assert
    # Result should be a dict with same keys
    assert isinstance(result, MultiHorizon)
    assert set(result.keys()) == set(sample_multi_horizon_data.keys())

    # Each horizon's data should have constant added to numeric columns
    for horizon, dataset in result.items():
        original_data = sample_multi_horizon_data[horizon].data
        expected_load = original_data["load"] + constant
        expected_temp = original_data["temperature"] + constant

        pd.testing.assert_series_equal(dataset.data["load"], expected_load, check_names=False)
        pd.testing.assert_series_equal(dataset.data["temperature"], expected_temp, check_names=False)
        assert dataset.sample_interval == sample_multi_horizon_data[horizon].sample_interval


def test_multi_horizon_transform_adapter_to_state_and_from_state(
    dummy_transform: DummyTransform, sample_multi_horizon_data: MultiHorizon[TimeSeriesDataset]
):
    """Test serialization and deserialization of the adapter."""
    # Arrange
    adapter = MultiHorizonTransformAdapter(time_series_transform=dummy_transform)
    adapter.fit(sample_multi_horizon_data)

    # Act
    state = adapter.to_state()
    new_adapter = MultiHorizonTransformAdapter(time_series_transform=DummyTransform())
    new_adapter.from_state(state)

    # Assert
    # State should contain the underlying transform's state
    state_dict = cast(dict[str, Any], state)
    assert "constant" in state_dict
    assert "fitted" in state_dict
    assert state_dict["constant"] == 5.0
    assert state_dict["fitted"]

    # Restored adapter should have same fitted state
    assert new_adapter.is_fitted
    assert cast(DummyTransform, new_adapter.time_series_transform).constant == 5.0
