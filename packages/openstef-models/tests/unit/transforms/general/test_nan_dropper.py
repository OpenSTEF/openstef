# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.transforms.general import NaNDropper
from openstef_models.utils.feature_selection import FeatureSelection


def test_nan_dropper__removes_rows_with_nan_in_selected_columns(caplog: pytest.LogCaptureFixture):
    """Test that NaNDropper removes rows containing NaN in selected columns and logs warning."""
    # Arrange
    data = pd.DataFrame(
        {
            "load": [100.0, np.nan, 110.0, 130.0, 140.0],
            "temperature": [20.0, 22.0, np.nan, 23.0, 24.0],
            "humidity": [60.0, 65.0, 70.0, np.nan, 80.0],
        },
        index=pd.date_range("2025-01-01", periods=5, freq="1h"),
    )
    dataset = TimeSeriesDataset(data, timedelta(hours=1))
    dropper = NaNDropper(selection=FeatureSelection(include={"load", "temperature"}))

    # Act
    transformed = dropper.transform(dataset)

    # Assert
    # Row 1 (index 1) has NaN in load, row 2 (index 2) has NaN in temperature - both should be dropped
    # Row 3 (index 3) has NaN in humidity but humidity is not selected, so it should remain
    # Remaining rows: 0, 3, 4
    expected_df = pd.DataFrame(
        {
            "load": [100.0, 130.0, 140.0],
            "temperature": [20.0, 23.0, 24.0],
            "humidity": [60.0, np.nan, 80.0],
        },
        index=pd.DatetimeIndex(["2025-01-01 00:00:00", "2025-01-01 03:00:00", "2025-01-01 04:00:00"], name="timestamp"),
    )
    pd.testing.assert_frame_equal(transformed.data, expected_df)
    assert transformed.sample_interval == dataset.sample_interval
    # 40% of rows dropped (2 out of 5), should trigger warning (default threshold is 10%)
    assert "NaNDropper dropped 2 of 5 rows (40.0%)" in caplog.text
