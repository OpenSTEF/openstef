# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pandas as pd
import pytest

from openstef_core.datasets import ForecastDataset
from openstef_models.transforms.postprocessing import QuantileSorter



def test_quantile_sorter_transform_enforces_monotonic_ordering():
    """Test that QuantileSorter correctly sorts quantiles to enforce monotonic ordering."""
    # Arrange
    sorter = QuantileSorter()
    unsorted_dataset = ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P10": [1.0, 2.0, 3.0],
                "quantile_P50": [0.5, 1.5, 2.5],  # Violates ordering
                "quantile_P90": [2.0, 3.0, 4.0],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )
    expected_sorted_data = pd.DataFrame(
        {
            "quantile_P10": [0.5, 1.5, 2.5],
            "quantile_P50": [1.0, 2.0, 3.0],
            "quantile_P90": [2.0, 3.0, 4.0],
        },
        index=unsorted_dataset.data.index,
    )

    # Act
    sorted_dataset = sorter.transform(unsorted_dataset)

    # Assert
    # Check that all rows have the expected sorted values
    pd.testing.assert_frame_equal(sorted_dataset.data, expected_sorted_data)

    # Check that sample interval and index are preserved
    assert sorted_dataset.sample_interval == unsorted_dataset.sample_interval
    pd.testing.assert_index_equal(sorted_dataset.data.index, unsorted_dataset.data.index)