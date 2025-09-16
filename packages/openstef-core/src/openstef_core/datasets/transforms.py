# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform base classes for time series data processing.

This module provides abstract base classes for implementing data transformations
on both simple and versioned time series datasets. Transforms follow the
scikit-learn pattern with separate fit and transform phases.
"""

from abc import abstractmethod

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.datasets.versioned_timeseries import VersionedTimeSeriesDataset
from openstef_core.types import LeadTime

type MultiHorizonTimeSeriesDataset = dict[LeadTime, TimeSeriesDataset]


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

    def fit_horizons(self, data: dict[LeadTime, TimeSeriesDataset]) -> None:
        """Fit the transform to multiple horizons of time series data.

        This method allows fitting the transform on a dictionary of datasets,
        each corresponding to a different lead time (horizon). It is useful
        for transforms that need to learn parameters across multiple horizons.

        Args:
            data: A dictionary mapping lead times to their corresponding
                  TimeSeriesDataset instances.
        """

    def transform_horizons(self, data: MultiHorizonTimeSeriesDataset) -> MultiHorizonTimeSeriesDataset:
        """Transform multiple horizons of time series data.

        This method applies the transformation to each dataset in the input
        dictionary, returning a new dictionary with the transformed datasets.

        Args:
            data: A dictionary mapping lead times to their corresponding
                  TimeSeriesDataset instances.

        Returns:
            A new dictionary mapping lead times to their transformed
            TimeSeriesDataset instances.
        """
        transformed_data: MultiHorizonTimeSeriesDataset = {}
        for horizon, dataset in data.items():
            transformed_data[horizon] = self.transform(dataset)

        return transformed_data

    def fit_transform_horizons(self, data: MultiHorizonTimeSeriesDataset) -> MultiHorizonTimeSeriesDataset:
        """Fit the transform to multiple horizons and then transform them.

        This method combines fitting and transforming for multiple horizons
        into a single step.

        Args:
            data: A dictionary mapping lead times to their corresponding
                  TimeSeriesDataset instances.

        Returns:
            A new dictionary mapping lead times to their transformed
            TimeSeriesDataset instances.
        """
        self.fit_horizons(data)
        return self.transform_horizons(data)


class VersionedTimeSeriesTransform:
    """Abstract base class for transforming versioned time series datasets.

    This class defines the interface for data transformations that operate on
    VersionedTimeSeriesDataset instances. Like TimeSeriesTransform, it follows
    the scikit-learn pattern but handles datasets with versioning information.

    Transforms on versioned datasets must preserve the timestamp and availability
    columns while transforming the feature data. This ensures that the versioning
    semantics are maintained through the transformation pipeline.

    Subclasses must implement the transform method and optionally override
    the fit method if the transformation requires learning parameters from data.

    Example:
        Implement a normalization transform for versioned data:

        >>> class VersionedNormalizeTransform(VersionedTimeSeriesTransform):
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
        ...         return VersionedTimeSeriesDataset(
        ...             normalized_data, data.sample_interval
        ...         )
    """

    def fit(self, data: VersionedTimeSeriesDataset) -> None:
        """Fit the transform to the input versioned time series data.

        This method should be called before applying the transform to the data.
        It allows the transform to learn any necessary parameters from the data.

        Args:
            data: The input versioned time series data to fit the transform on.
        """

    @abstractmethod
    def transform(self, data: VersionedTimeSeriesDataset) -> VersionedTimeSeriesDataset:
        """Transform the input versioned time series data.

        This method should apply a transformation to the input data and return
        a new instance of VersionedTimeSeriesDataset.

        Args:
            data: The input versioned time series data to be transformed.

        Returns:
            A new instance of VersionedTimeSeriesDataset containing the
            transformed data.
        """
        raise NotImplementedError

    def fit_transform(self, data: VersionedTimeSeriesDataset) -> VersionedTimeSeriesDataset:
        """Fit the transform to the data and then transform it.

        This method combines fitting and transforming into a single step.

        Args:
            data: The input versioned time series data to fit and transform.

        Returns:
            A new instance of VersionedTimeSeriesDataset containing the transformed data.
        """
        self.fit(data)
        return self.transform(data)


class ForecastTransform:
    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, data: ForecastDataset) -> None:
        pass

    @abstractmethod
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        raise NotImplementedError

    def fit_transform(self, data: ForecastDataset) -> ForecastDataset:
        self.fit(data)
        return self.transform(data)


__all__ = [
    "ForecastTransform",
    "TimeSeriesTransform",
    "VersionedTimeSeriesTransform",
]
