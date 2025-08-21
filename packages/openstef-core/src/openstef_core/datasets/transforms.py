# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform base classes for time series data processing.

This module provides abstract base classes for implementing data transformations
on both simple and versioned time series datasets. Transforms follow the
scikit-learn pattern with separate fit and transform phases.
"""

from abc import abstractmethod

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeseriesDataset


class TimeSeriesTransform:
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
        ...         self.scale_factor = 1.0
        ...
        ...     def fit(self, data):
        ...         self.scale_factor = data.data.max().max()
        ...
        ...     def transform(self, data):
        ...         scaled_data = data.data / self.scale_factor
        ...         return TimeSeriesDataset(scaled_data, data.sample_interval)
    """

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the input time series data.

        This method should be called before applying the transform to the data.
        It allows the transform to learn any necessary parameters from the data.

        Args:
            data: The input time series data to fit the transform on.
        """

    @abstractmethod
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data.

        This method should apply a transformation to the input data and return a new instance of TimeSeriesDataset.

        Args:
            data: The input time series data to be transformed.

        Returns:
            A new instance of TimeSeriesDataset containing the transformed data.
        """
        raise NotImplementedError

    def fit_transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Fit the transform to the data and then transform it.

        This method combines fitting and transforming into a single step.

        Args:
            data: The input time series data to fit and transform.

        Returns:
            A new instance of TimeSeriesDataset containing the transformed data.
        """
        self.fit(data)
        return self.transform(data)


class TimeSeriesVersionedTransform:
    """Abstract base class for transforming versioned time series datasets.

    This class defines the interface for data transformations that operate on
    VersionedTimeseriesDataset instances. Like TimeSeriesTransform, it follows
    the scikit-learn pattern but handles datasets with versioning information.

    Transforms on versioned datasets must preserve the timestamp and availability
    columns while transforming the feature data. This ensures that the versioning
    semantics are maintained through the transformation pipeline.

    Subclasses must implement the transform method and optionally override
    the fit method if the transformation requires learning parameters from data.

    Example:
        Implement a normalization transform for versioned data:

        >>> class VersionedNormalizeTransform(TimeSeriesVersionedTransform):
        ...     def __init__(self):
        ...         self.mean = None
        ...         self.std = None
        ...
        ...     def fit(self, data):
        ...         feature_data = data.data[data.feature_names]
        ...         self.mean = feature_data.mean()
        ...         self.std = feature_data.std()
        ...
        ...     def transform(self, data):
        ...         normalized_data = data.data.copy()
        ...         normalized_data[data.feature_names] = (
        ...             (data.data[data.feature_names] - self.mean) / self.std
        ...         )
        ...         return VersionedTimeseriesDataset(
        ...             normalized_data, data.sample_interval
        ...         )
    """

    def fit(self, data: VersionedTimeseriesDataset) -> None:
        """Fit the transform to the input versioned time series data.

        This method should be called before applying the transform to the data.
        It allows the transform to learn any necessary parameters from the data.

        Args:
            data: The input versioned time series data to fit the transform on.
        """

    @abstractmethod
    def transform(self, data: VersionedTimeseriesDataset) -> VersionedTimeseriesDataset:
        """Transform the input versioned time series data.

        This method should apply a transformation to the input data and return
        a new instance of VersionedTimeseriesDataset.

        Args:
            data: The input versioned time series data to be transformed.

        Returns:
            A new instance of VersionedTimeseriesDataset containing the
            transformed data.
        """
        raise NotImplementedError

    def fit_transform(self, data: VersionedTimeseriesDataset) -> VersionedTimeseriesDataset:
        """Fit the transform to the data and then transform it.

        This method combines fitting and transforming into a single step.

        Args:
            data: The input versioned time series data to fit and transform.

        Returns:
            A new instance of VersionedTimeseriesDataset containing the transformed data.
        """
        self.fit(data)
        return self.transform(data)


__all__ = [
    "TimeSeriesTransform",
    "TimeSeriesVersionedTransform",
]
