# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import functools
import operator
from collections import Counter

from openstef_beam.datasets.base import TimeseriesDataset


def check_features_are_disjoint(datasets: list[TimeseriesDataset]) -> None:
    """Check if the datasets have overlapping feature names.
    If they do, raise a ValueError.
    """
    all_features: list[str] = functools.reduce(operator.iadd, [d.feature_names for d in datasets], [])
    if len(all_features) != len(set(all_features)):
        duplicate_features = [item for item, count in Counter(all_features).items() if count > 1]
        raise ValueError("Datasets have overlapping feature names: " + ", ".join(duplicate_features))


def check_sample_intervals(datasets: list[TimeseriesDataset]) -> None:
    """Check if the datasets have the same sample interval.
    If they do not, raise a ValueError.
    """
    sample_intervals = {d.sample_interval for d in datasets}
    if len(sample_intervals) > 1:
        raise ValueError("Datasets have different sample intervals: " + ", ".join(map(str, sample_intervals)))
