# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""New transform interfaces for time series data processing.

This module provides updated abstract base classes for implementing data transformations
on time series datasets with support for multi-horizon and versioned datasets.
These interfaces build on the core Transform mixin with specialized support for
forecasting use cases.
"""

from abc import abstractmethod
from typing import override

from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.mixins import Transform


class TimeSeriesTransform(Transform[TimeSeriesDataset, TimeSeriesDataset]):
    """Abstract base class for transforming regular time series datasets.

    This class defines the interface for data transformations that operate on
    TimeSeriesDataset instances. Transforms follow the scikit-learn pattern
    with separate fit and transform phases, allowing for stateful transformations
    that learn parameters from training data.

    Subclasses must implement the transform method and optionally override
    the fit method if the transformation requires learning parameters from data.

    Example:
        Implement a simple scaling transform:

        >>> class ScaleTransform(TimeSeriesTransform):
        ...     def __init__(self):
        ...         self.scale_factor = None
        ...
        ...     @property
        ...     def is_fitted(self) -> bool:
        ...         return self.scale_factor is not None
        ...
        ...     def fit(self, data):
        ...         self.scale_factor = data.data.max().max()
        ...
        ...     def transform(self, data):
        ...         scaled_data = data.data / self.scale_factor
        ...         return TimeSeriesDataset(scaled_data, data.sample_interval)
    """

    # Stateless by default
    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        pass

    @abstractmethod
    def features_added(self) -> list[str]:
        """List of feature names added by this transform.

        Returns:
            A list of strings representing the names of features added
            to the dataset by this transform. Default is an empty list.
        """


class VersionedTimeSeriesTransform(Transform[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]):
    """Abstract base class for transforming versioned time series datasets.

    This class provides an interface for transformations that operate on
    VersionedTimeSeriesDataset instances, which include time point versions
    enabling real life data availability simulation.

    Subclasses must implement the fit and transform methods.
    """


__all__ = ["TimeSeriesTransform", "VersionedTimeSeriesTransform"]
