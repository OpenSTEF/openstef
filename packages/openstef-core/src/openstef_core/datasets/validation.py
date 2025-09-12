# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Validation utilities for time series datasets.

This module provides functions to validate dataset compatibility and integrity,
particularly for operations that combine multiple datasets.
"""

import functools
import operator
from collections import Counter
from collections.abc import Iterable
from datetime import timedelta

import pandas as pd

from openstef_core.datasets.mixins import TimeSeriesMixin
from openstef_core.exceptions import InvalidColumnTypeError, MissingColumnsError, TimeSeriesValidationError


def validate_required_columns(dataset: TimeSeriesMixin, required_columns: list[str]) -> None:
    """Check if the dataset contains all required columns.

    Validates that the dataset includes all specified required columns,
    raising an error if any are missing.

    Args:
        dataset: The time series dataset to validate.
        required_columns: List of column names that must be present in the dataset.

    Raises:
        MissingColumnsError: If any required columns are missing from the dataset.
    """
    missing_columns = [col for col in required_columns if col not in dataset.feature_names]
    if missing_columns:
        raise MissingColumnsError(missing_columns=missing_columns)


def validate_disjoint_columns(datasets: Iterable[TimeSeriesMixin]) -> None:
    """Check if the datasets have overlapping feature names.

    Validates that all datasets have completely disjoint feature sets,
    ensuring no feature appears in multiple datasets.

    Args:
        datasets: Sequence of time series datasets to validate.

    Raises:
        TimeSeriesValidationError: If any feature name appears in multiple datasets.
    """
    all_features: list[str] = functools.reduce(operator.iadd, [d.feature_names for d in datasets], [])
    if len(all_features) != len(set(all_features)):
        duplicate_features = [item for item, count in Counter(all_features).items() if count > 1]
        raise TimeSeriesValidationError("Datasets have overlapping feature names: " + ", ".join(duplicate_features))


def validate_same_sample_intervals(datasets: Iterable[TimeSeriesMixin]) -> timedelta:
    """Check if the datasets have the same sample interval.

    Validates that all datasets use identical sampling intervals, which is
    required for safe concatenation and combination operations.

    Args:
        datasets: Sequence of time series datasets to validate.

    Returns:
        The common sample interval shared by all datasets.

    Raises:
        TimeSeriesValidationError: If datasets have different sample intervals.
    """
    sample_intervals = {d.sample_interval for d in datasets}
    if len(sample_intervals) > 1:
        raise TimeSeriesValidationError(
            "Datasets have different sample intervals: " + ", ".join(map(str, sample_intervals))
        )

    return sample_intervals.pop()


def validate_datetime_column(series: pd.Series, column_name: str) -> None:
    """Validate that a pandas Series is of datetime type.

    Args:
        series: The pandas Series to validate.
        column_name: Name of the column being validated (for error messages).

    Raises:
        InvalidColumnTypeError: If the series is not of datetime type.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise InvalidColumnTypeError(column_name, expected_type="datetime", actual_type=str(series.dtype))
