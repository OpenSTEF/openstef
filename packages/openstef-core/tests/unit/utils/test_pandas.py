# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pandas utilities."""

import numpy as np
import pandas as pd
import pytest

from openstef_core.utils.pandas import nan_aware_weighted_mean


@pytest.mark.parametrize(
    ("values", "weights", "expected"),
    [
        pytest.param(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            {"a": [0.6, 0.5], "b": [0.4, 0.5]},
            [1.8, 3.0],
            id="no_nans",
        ),
        pytest.param(
            {"a": [1.0, np.nan], "b": [3.0, 4.0]},
            {"a": [0.6, 0.6], "b": [0.4, 0.4]},
            [1.8, 4.0],
            id="nan_redistributes_weight",
        ),
        pytest.param(
            {"a": [np.nan], "b": [np.nan]},
            {"a": [0.5], "b": [0.5]},
            [0.0],
            id="all_nan_returns_zero",
        ),
        pytest.param(
            {"a": [5.0, np.nan, 3.0]},
            {"a": [1.0, 1.0, 1.0]},
            [5.0, 0.0, 3.0],
            id="single_column",
        ),
    ],
)
def test_nan_aware_weighted_mean(
    values: dict[str, list[float]],
    weights: dict[str, list[float]],
    expected: list[float],
) -> None:
    # Arrange
    values_df = pd.DataFrame(values)
    weights_df = pd.DataFrame(weights)

    # Act
    result = nan_aware_weighted_mean(values_df, weights_df)

    # Assert
    pd.testing.assert_series_equal(result, pd.Series(expected), check_names=False)
