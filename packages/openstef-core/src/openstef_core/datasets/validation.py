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
from collections.abc import Sequence

from openstef_core.datasets.mixins import TimeSeriesMixin
from openstef_core.exceptions import TimeSeriesValidationError


def check_features_are_disjoint(datasets: Sequence[TimeSeriesMixin]) -> None:
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


def check_sample_intervals(datasets: Sequence[TimeSeriesMixin]) -> None:
    """Check if the datasets have the same sample interval.

    Validates that all datasets use identical sampling intervals, which is
    required for safe concatenation and combination operations.

    Args:
        datasets: Sequence of time series datasets to validate.

    Raises:
        TimeSeriesValidationError: If datasets have different sample intervals.
    """
    sample_intervals = {d.sample_interval for d in datasets}
    if len(sample_intervals) > 1:
        raise TimeSeriesValidationError(
            "Datasets have different sample intervals: " + ", ".join(map(str, sample_intervals))
        )
