# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import override

import pandas as pd
import pytest

from openstef_core.datasets.transforms import ForecastTransform
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_models.transforms.forecast_transform_pipeline import ForecastTransformPipeline


class SimpleAddTransform(ForecastTransform):
    """Simple transform that adds a constant to all forecast values."""

    def __init__(self, add_value: float = 1.0):
        self.add_value = add_value

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        return ForecastDataset(
            data=data.data + self.add_value,
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
        )


class SimpleMultiplyTransform(ForecastTransform):
    """Simple transform that multiplies all forecast values by a constant."""

    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        return ForecastDataset(
            data=data.data * self.multiplier,
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
        )


@pytest.fixture
def sample_forecast_dataset() -> ForecastDataset:
    """Create a sample ForecastDataset for testing."""
    data = pd.DataFrame(
        {
            "quantile_P10": [90.0, 95.0],
            "quantile_P50": [100.0, 105.0],
            "quantile_P90": [110.0, 115.0],
        },
        index=pd.date_range("2025-01-01 10:00", periods=2, freq="h"),
    )

    return ForecastDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        forecast_start=datetime.fromisoformat("2025-01-01T10:00:00"),
    )


@pytest.mark.parametrize(
    ("transforms", "expected_values"),
    [
        pytest.param([], [90.0, 95.0], id="no_transforms"),
        pytest.param([SimpleAddTransform(5.0)], [95.0, 100.0], id="single_transform"),
        pytest.param([SimpleAddTransform(10.0), SimpleMultiplyTransform(2.0)], [200.0, 210.0], id="chained_transforms"),
    ],
)
def test_postprocessing_pipeline__fit_transform_functionality(
    sample_forecast_dataset: ForecastDataset,
    transforms: list[ForecastTransform],
    expected_values: list[float],
):
    """Test PostprocessingPipeline fit_transform with essential scenarios."""
    # Arrange
    pipeline = ForecastTransformPipeline(transforms=transforms)

    # Act - fit is required before transform
    result = pipeline.fit_transform(sample_forecast_dataset)

    # Assert
    assert result.data["quantile_P10"].tolist() == expected_values
    assert result.sample_interval == sample_forecast_dataset.sample_interval
    assert result.forecast_start == sample_forecast_dataset.forecast_start
