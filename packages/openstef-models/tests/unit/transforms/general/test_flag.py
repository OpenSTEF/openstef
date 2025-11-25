# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pytest

from openstef_models.transforms.general.flag import compute_range_flag


def test_missing_features_raise_error():
    df = pd.DataFrame({"a": [1], "b": [2]})
    input_features = pd.DataFrame({"a": [1], "b": [2]})
    features_to_flag = ["a", "c"]  # "c" is missing

    with pytest.raises(ValueError) as exc_info:
        compute_range_flag(
            df,
            input_features,
            features_to_flag,
            feature_ranges={"a": (0, 2)}
        )

    # Convert the set to a string to match the output formatting
    assert "Features to flag not found in input DataFrame: {'c'}" in str(exc_info.value)


def test_all_values_inside_range():
    """test wether all values are inside the range and correctly put to 1 if inside of range."""
    df = pd.DataFrame(index=[0, 1])
    input_features = pd.DataFrame({"a": [5, 7]}, index=[0, 1])
    features_to_flag = ["a"]
    feature_ranges = {"a": (0, 10)}

    result = compute_range_flag(df, input_features, features_to_flag, feature_ranges) # type: ignore
    assert list(result) == [1, 1]

def test_values_outside_range():
    """test wether values outside the range are correctly put to 0."""
    df = pd.DataFrame(index=[0, 1, 2])
    input_features = pd.DataFrame({"a": [5, 20, -3]}, index=[0, 1, 2])
    features_to_flag = ["a"]
    feature_ranges = {"a": (0, 10)}

    result = compute_range_flag(df, input_features, features_to_flag, feature_ranges) # type: ignore
    assert list(result) == [1, 0, 0]  # only first row is inside