# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import InsufficientlyCompleteError
from openstef_core.feature_engineering.validation_transforms.completeness_check import (
    CompletenessCheckTransform,
)


# @pytest.mark.parametrize(
#     ("columns", "weights", "threshold", "completeness", "sufficiently_complete"),
#     [
#         pytest.param(
#             None,
#             {"radiation": 1.0, "temperature": 1.0, "wind_speed": 1.0},
#             0.5,
#             0.75,
#             True,
#             id="sufficient_equal_weights",
#         ),
#         pytest.param(None, None, 0.5, 0.75, True, id="sufficient_default_weights"),
#         pytest.param(
#             None,
#             {"radiation": 1.0, "temperature": 1.0, "wind_speed": 1.0},
#             0.8,
#             0.75,
#             False,
#             id="insufficient_equal_weights",
#         ),
#         pytest.param(
#             None,
#             {"radiation": 1.0, "temperature": 3.0, "wind_speed": 1.0},
#             0.5,
#             0.75,
#             True,
#             id="sufficient_unequal_weights",
#         ),
#         pytest.param(
#             ["radiation", "temperature"],
#             {"radiation": 1.0, "temperature": 1.0},
#             0.5,
#             0.75,
#             True,
#             id="sufficient_partial_columns",
#         ),
#     ],
# )
# def test_transform(
#     columns: list[str] | None,
#     weights: dict[str, float],
#     threshold: float,
#     completeness: float,
#     sufficiently_complete: bool,
# ):
#     data = pd.DataFrame(
#         {
#             "radiation": [100, 110, 110, np.nan],
#             "temperature": [20, np.nan, np.nan, 21],
#             "wind_speed": [5, 6, 6, 3],
#         },
#         index=pd.date_range("2025-01-01", periods=4, freq="15min"),
#     )
#     dataset = TimeSeriesDataset(data, timedelta(minutes=15))
#     transform = CompletenessCheckTransform(
#         columns=columns, weights=weights, completeness_threshold=threshold, error_on_insufficient_completeness=False
#     )
#     if not sufficiently_complete:
#         with pytest.raises(InsufficientlyCompleteError, match=f"The dataset is not sufficiently complete. Completeness: {completeness}"):
#             transform.transform(dataset)
#     else:
#         result = transform.transform(dataset)
#         assert result == dataset


@pytest.mark.parametrize(
    ("data_dict", "columns", "weights", "threshold", "expected_completeness"),
    [
        pytest.param(
            {
                "radiation": [100, 110, 110, np.nan],
                "temperature": [20, np.nan, np.nan, 21],
                "wind_speed": [5, 6, 6, 3],
            },
            None,
            None,
            0.5,
            0.75,
            id="mixed_completeness_default",
        ),
        pytest.param(
            {
                "radiation": [100, 110, 110, 120],
                "temperature": [20, 21, 22, 23],
                "wind_speed": [5, 6, 7, 8],
            },
            None,
            None,
            0.5,
            1.0,
            id="fully_complete",
        ),
        pytest.param(
            {
                "radiation": [100, np.nan, np.nan, np.nan],
                "temperature": [20, 21, 22, 23],
                "wind_speed": [5, 6, 7, 8],
            },
            ["temperature", "wind_speed"],
            None,
            0.5,
            1.0,
            id="complete_selected_columns",
        ),
        pytest.param(
            {
                "radiation": [100, 110, 110, 120],
                "temperature": [np.nan, np.nan, np.nan, np.nan],
                "wind_speed": [5, 6, 7, 8],
            },
            None,
            {"radiation": 2.0, "temperature": 1.0, "wind_speed": 2.0},
            0.5,
            0.8,
            id="weighted_completeness_sufficient",
        ),
        pytest.param(
            {
                "single_col": [100, np.nan, 110, np.nan],
            },
            None,
            None,
            0.5,
            0.5,
            id="single_column_exact_threshold",
        ),
    ],
)
def test_transform_sufficient_completeness(
    data_dict: dict[str, list[float]],
    columns: list[str] | None,
    weights: dict[str, float] | None,
    threshold: float,
    expected_completeness: float,
):
    data = pd.DataFrame(
        data_dict,
        index=pd.date_range("2025-01-01", periods=len(next(iter(data_dict.values()))), freq="15min"),
    )
    
    dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    transform = CompletenessCheckTransform(
        columns=columns, 
        weights=weights, 
        completeness_threshold=threshold,
        error_on_insufficient_completeness=True
    )
    
    result = transform.transform(dataset)
    assert result == dataset
    assert abs(transform._completeness - expected_completeness) < 0.01


@pytest.mark.parametrize(
    ("data_dict", "columns", "weights", "threshold", "expected_completeness"),
    [
        pytest.param(
            {
                "radiation": [np.nan, np.nan, np.nan, np.nan],
                "temperature": [np.nan, np.nan, np.nan, np.nan],
                "wind_speed": [np.nan, np.nan, np.nan, np.nan],
            },
            None,
            None,
            0.5,
            0.0,
            id="fully_missing",
        ),
        pytest.param(
            {
                "radiation": [100, np.nan, np.nan, np.nan],
                "temperature": [20, np.nan, np.nan, np.nan],
                "wind_speed": [5, np.nan, np.nan, np.nan],
            },
            None,
            None,
            0.5,
            0.25,
            id="mostly_missing",
        ),
        pytest.param(
            {
                "radiation": [100, 110, 110, 120],
                "temperature": [np.nan, np.nan, np.nan, np.nan],
                "wind_speed": [5, 6, 7, 8],
            },
            None,
            {"radiation": 1.0, "temperature": 10.0, "wind_speed": 1.0},
            0.5,
            0.17,
            id="weighted_completeness_insufficient",
        ),
        pytest.param(
            {},
            None,
            None,
            0.5,
            0.0,
            id="empty_dataframe",
        ),
        pytest.param(
            {
                "radiation": [100, 110, 110, np.nan],
                "temperature": [20, np.nan, np.nan, 21],
                "wind_speed": [5, 6, 6, 3],
            },
            None,
            {"radiation": 1.0, "temperature": 1.0, "wind_speed": 1.0},
            0.8,
            0.75,
            id="insufficient_equal_weights",
        ),
    ],
)
def test_transform_insufficient_completeness(
    data_dict: dict[str, list[float]],
    columns: list[str] | None,
    weights: dict[str, float] | None,
    threshold: float,
    expected_completeness: float,
):
    if data_dict:
        data = pd.DataFrame(
            data_dict,
            index=pd.date_range("2025-01-01", periods=len(next(iter(data_dict.values()))), freq="15min"),
        )
    else:
        data = pd.DataFrame(index=pd.date_range("2025-01-01", periods=0, freq="15min"))
    
    dataset = TimeSeriesDataset(data, timedelta(minutes=15))
    transform = CompletenessCheckTransform(
        columns=columns, 
        weights=weights, 
        completeness_threshold=threshold,
        error_on_insufficient_completeness=True
    )

    with pytest.raises(InsufficientlyCompleteError, match=f"The dataset is not sufficiently complete. Completeness: {expected_completeness}"):
        transform.transform(dataset)
