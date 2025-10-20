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
from collections.abc import Iterable, Sequence
from datetime import timedelta

import pandas as pd

from openstef_core.datasets.mixins import TimeSeriesMixin
from openstef_core.exceptions import InvalidColumnTypeError, MissingColumnsError, TimeSeriesValidationError


def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Check if the dataset contains all required columns.

    Validates that the dataset includes all specified required columns,
    raising an error if any are missing.

    Args:
        df: The time series dataset to validate.
        required_columns: List of column names that must be present in the dataset.

    Raises:
        MissingColumnsError: If any required columns are missing from the dataset.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise MissingColumnsError(missing_columns=missing_columns)


def validate_disjoint_columns(datasets: Iterable[TimeSeriesMixin]) -> list[str]:
    """Check if the datasets have overlapping feature names.

    Validates that all datasets have completely disjoint feature sets,
    ensuring no feature appears in multiple datasets.

    Args:
        datasets: Sequence of time series datasets to validate.

    Returns:
        The combined list of all feature names across the datasets.

    Raises:
        TimeSeriesValidationError: If any feature name appears in multiple datasets.
    """
    all_features: list[str] = functools.reduce(operator.iadd, [d.feature_names for d in datasets], [])
    if len(all_features) != len(set(all_features)):
        duplicate_features = [item for item, count in Counter(all_features).items() if count > 1]
        raise TimeSeriesValidationError("Datasets have overlapping feature names: " + ", ".join(duplicate_features))

    return all_features


def validate_same_columns(datasets: Iterable[TimeSeriesMixin]) -> list[str]:
    """Check if the datasets have the same feature names.

    Validates that all datasets contain identical sets of feature names,
    which is required for safe concatenation and combination operations.

    Args:
        datasets: Sequence of time series datasets to validate.

    Returns:
        The common list of feature names shared by all datasets.

    Raises:
        TimeSeriesValidationError: If datasets have different feature names.
    """
    feature_sets = {frozenset(d.feature_names) for d in datasets}
    if len(feature_sets) > 1:
        raise TimeSeriesValidationError(
            "Datasets have different feature names: " + "; ".join([", ".join(sorted(fs)) for fs in feature_sets])
        )

    return list(feature_sets.pop())


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


def validate_datetime_column(series: pd.Series, column_name: str | None = None) -> None:
    """Validate that a pandas Series is of datetime type.

    Args:
        series: The pandas Series to validate.
        column_name: Name of the column being validated (for error messages).

    Raises:
        InvalidColumnTypeError: If the series is not of datetime type.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise InvalidColumnTypeError(
            str(series.name or column_name), expected_type="datetime", actual_type=str(series.dtype)
        )


__all__ = [
    "validate_datetime_column",
    "validate_disjoint_columns",
    "validate_required_columns",
    "validate_same_columns",
    "validate_same_sample_intervals",
]
