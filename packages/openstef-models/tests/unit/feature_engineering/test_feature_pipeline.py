# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for FeaturePipeline and related transform orchestration."""

from datetime import timedelta
from typing import override

import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform, VersionedTimeSeriesTransform
from openstef_core.datasets.versioned_timeseries.dataset_part import VersionedTimeSeriesPart
from openstef_core.exceptions import TransformNotFittedError
from openstef_core.types import LeadTime
from openstef_models.feature_engineering.feature_pipeline import FeaturePipeline
from openstef_models.feature_engineering.horizon_split_transform import concat_horizon_datasets_rowwise


class MockTimeSeriesTransform(TimeSeriesTransform):
    """Mock TimeSeriesTransform for testing."""

    def __init__(self, transform_suffix: str = "_transformed"):
        self.transform_suffix = transform_suffix
        self.is_fitted = False

    @override
    def fit_horizons(self, data: dict[LeadTime, TimeSeriesDataset]) -> None:
        flat_data = concat_horizon_datasets_rowwise(data)
        return self.fit(flat_data)

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        self.is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self.is_fitted:
            raise TransformNotFittedError("Mock transform not fitted")

        # Add suffix to column names
        new_data = data.data.copy()
        new_data.columns = [f"{col}{self.transform_suffix}" for col in new_data.columns]
        return TimeSeriesDataset(new_data, data.sample_interval)


class MockVersionedTimeSeriesTransform(VersionedTimeSeriesTransform):
    """Mock VersionedTimeSeriesTransform for testing."""

    def __init__(self, transform_suffix: str = "_versioned"):
        self.transform_suffix = transform_suffix
        self.is_fitted = False

    @override
    def fit(self, data: VersionedTimeSeriesDataset) -> None:
        self.is_fitted = True

    @override
    def transform(self, data: VersionedTimeSeriesDataset) -> VersionedTimeSeriesDataset:
        if not self.is_fitted:
            raise TransformNotFittedError("Mock versioned transform not fitted")

        # Add suffix to feature names in data parts
        new_parts: list[VersionedTimeSeriesPart] = []
        for part in data.data_parts:
            new_data = part.data.rename(
                columns={
                    col: f"{col}{self.transform_suffix}"
                    for col in part.data.columns
                    if col not in {"timestamp", "available_at"}
                }
            )
            # Create new part with transformed data
            new_part = VersionedTimeSeriesPart(new_data, part.sample_interval)
            new_parts.append(new_part)

        return VersionedTimeSeriesDataset(new_parts)


@pytest.fixture
def sample_versioned_dataset() -> VersionedTimeSeriesDataset:
    """Create sample versioned dataset for testing."""
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01T10:00:00",
            "2025-01-01T11:00:00",
            "2025-01-01T12:00:00",
        ]),
        "available_at": pd.to_datetime([
            "2025-01-01T10:05:00",
            "2025-01-01T11:05:00",
            "2025-01-01T12:05:00",
        ]),
        "load": [100.0, 110.0, 120.0],
        "temperature": [20.0, 21.0, 22.0],
    })
    return VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample TimeSeriesDataset for testing."""
    data = pd.DataFrame(
        {
            "load": [100.0, 110.0, 120.0],
            "temperature": [20.0, 21.0, 22.0],
        },
        index=pd.date_range("2025-01-01 10:00", periods=3, freq="1h"),
    )
    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.mark.parametrize(
    ("horizons", "versioned_transforms", "horizon_transforms", "expected_columns"),
    [
        pytest.param([LeadTime.from_string("PT1H")], [], [], {"load", "temperature"}, id="no_transforms"),
        pytest.param(
            [LeadTime.from_string("PT1H")],
            [MockVersionedTimeSeriesTransform("_v")],
            [MockTimeSeriesTransform("_h")],
            {"load_v_h", "temperature_v_h"},
            id="single_horizon_all_transforms",
        ),
        pytest.param(
            [LeadTime.from_string("PT1H"), LeadTime.from_string("PT24H")],
            [MockVersionedTimeSeriesTransform("_v")],
            [MockTimeSeriesTransform("_h")],
            {"load_v_h", "temperature_v_h"},
            id="multiple_horizons_with_transforms",
        ),
        pytest.param(
            [LeadTime.from_string("PT1H")],
            [MockVersionedTimeSeriesTransform("_v1"), MockVersionedTimeSeriesTransform("_v2")],
            [MockTimeSeriesTransform("_h1"), MockTimeSeriesTransform("_h2")],
            {"load_v1_v2_h1_h2", "temperature_v1_v2_h1_h2"},
            id="multiple_transforms_chained",
        ),
    ],
)
def test_feature_pipeline_core_functionality(
    sample_versioned_dataset: VersionedTimeSeriesDataset,
    horizons: list[LeadTime],
    versioned_transforms: list[VersionedTimeSeriesTransform],
    horizon_transforms: list[TimeSeriesTransform],
    expected_columns: set[str],
):
    """Test core FeaturePipeline functionality with essential scenarios."""
    # Arrange
    pipeline = FeaturePipeline(
        horizons=horizons, versioned_transforms=versioned_transforms, horizon_transforms=horizon_transforms
    )

    # Act
    result = pipeline.fit_transform(sample_versioned_dataset)

    # Assert
    assert pipeline._is_fitted
    assert len(result) == len(horizons)
    for horizon in horizons:
        assert horizon in result
        horizon_dataset = result[horizon]
        assert isinstance(horizon_dataset, TimeSeriesDataset)
        assert set(horizon_dataset.data.columns) == expected_columns


def test_feature_pipeline_transform_not_fitted_error(sample_versioned_dataset: VersionedTimeSeriesDataset):
    """Test that transform raises error when pipeline not fitted."""
    # Arrange
    pipeline = FeaturePipeline()

    # Act & Assert
    with pytest.raises(TransformNotFittedError, match="Pipeline is not fitted yet"):
        pipeline.transform(sample_versioned_dataset)


def test_feature_pipeline_unversioned_dataset_compatibility(sample_timeseries_dataset: TimeSeriesDataset):
    """Test pipeline works with unversioned dataset when compatible."""
    # Arrange
    pipeline = FeaturePipeline(
        horizons=[LeadTime.from_string("PT1H")],
        versioned_transforms=[],
        horizon_transforms=[MockTimeSeriesTransform("_test")],
    )

    # Act
    pipeline.fit(sample_timeseries_dataset)
    result = pipeline.transform(sample_timeseries_dataset)

    # Assert
    assert pipeline._is_fitted
    assert len(result) == 1
    horizon_dataset = result[LeadTime.from_string("PT1H")]
    assert set(horizon_dataset.data.columns) == {"load_test", "temperature_test"}


@pytest.mark.parametrize(
    ("horizon_transforms", "expected_columns"),
    [
        pytest.param([], {"load", "temperature"}, id="no_transforms"),
        pytest.param(
            [MockTimeSeriesTransform("_h")],
            {"load_h", "temperature_h"},
            id="single_transform",
        ),
        pytest.param(
            [MockTimeSeriesTransform("_h1"), MockTimeSeriesTransform("_h2")],
            {"load_h1_h2", "temperature_h1_h2"},
            id="multiple_transforms_chained",
        ),
    ],
)
def test_feature_pipeline_unversioned_dataset_scenarios(
    sample_timeseries_dataset: TimeSeriesDataset,
    horizon_transforms: list[TimeSeriesTransform],
    expected_columns: set[str],
):
    """Test FeaturePipeline with unversioned datasets across different transform scenarios."""
    # Arrange
    pipeline = FeaturePipeline(
        horizons=[LeadTime.from_string("PT1H")],
        versioned_transforms=[],
        horizon_transforms=horizon_transforms,
    )

    # Act
    result = pipeline.fit_transform(sample_timeseries_dataset)

    # Assert
    assert pipeline._is_fitted
    assert len(result) == 1
    horizon_dataset = result[LeadTime.from_string("PT1H")]
    assert isinstance(horizon_dataset, TimeSeriesDataset)
    assert set(horizon_dataset.data.columns) == expected_columns


def test_feature_pipeline_unversioned_dataset_validation_errors(sample_timeseries_dataset: TimeSeriesDataset):
    """Test validation errors for incompatible unversioned dataset configurations."""
    # Arrange - Pipeline with versioned transforms
    pipeline_with_versioned = FeaturePipeline(
        horizons=[LeadTime.from_string("PT1H")], versioned_transforms=[MockVersionedTimeSeriesTransform()]
    )

    # Act & Assert
    with pytest.raises(ValueError, match="pipeline cannot contain versioned transforms"):
        pipeline_with_versioned.fit(sample_timeseries_dataset)

    # Arrange - Pipeline with multiple horizons
    pipeline_multi_horizon = FeaturePipeline(horizons=[LeadTime.from_string("PT1H"), LeadTime.from_string("PT24H")])

    # Act & Assert
    with pytest.raises(ValueError, match="exactly one horizon must be configured"):
        pipeline_multi_horizon.fit(sample_timeseries_dataset)
