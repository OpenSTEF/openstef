# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Time series data transformation utilities for forecasting.

This module provides abstract base classes and utilities for transforming
time series datasets, with support for multi-horizon forecasting scenarios.
"""

from abc import ABC, abstractmethod
from typing import override

import pandas as pd

from openstef_core.datasets import MultiHorizonTimeSeriesDataset, TimeSeriesDataset
from openstef_core.datasets.transforms import SelfTransform
from openstef_core.types import LeadTime


class MultiHorizonTransform(ABC):
    """Abstract base class for transforms that operate on multiple forecast horizons.

    This class provides an interface for data transformations that need to handle
    multiple forecast horizons simultaneously, such as feature engineering that
    requires learning parameters across different lead times.
    """

    @abstractmethod
    def fit_horizons(self, data: MultiHorizonTimeSeriesDataset) -> None:
        """Fit the transform to multiple horizons of time series data.

        This method allows fitting the transform on a dictionary of datasets,
        each corresponding to a different lead time (horizon). It is useful
        for transforms that need to learn parameters across multiple horizons.

        Args:
            data: A dictionary mapping lead times to their corresponding
                  TimeSeriesDataset instances.
        """

    @abstractmethod
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


def concat_horizon_datasets_rowwise(horizon_datasets: dict[LeadTime, TimeSeriesDataset]) -> TimeSeriesDataset:
    """Concatenate multiple horizon datasets into a single dataset.

    This function takes a dictionary of horizon datasets, each corresponding to a specific lead time,
    and concatenates them row-wise into a single TimeSeriesDataset. The resulting dataset contains all
    the data from the input datasets, with the same sample interval as the first dataset in the input.

    Args:
        horizon_datasets: A dictionary where keys are LeadTime objects and values are TimeSeriesDataset instances.

    Returns:
        A single TimeSeriesDataset containing all rows from the input datasets.
    """
    return TimeSeriesDataset(
        data=pd.concat([horizon_data.data for horizon_data in horizon_datasets.values()]),
        sample_interval=horizon_datasets[next(iter(horizon_datasets))].sample_interval,
    )


class TimeSeriesTransform(SelfTransform[TimeSeriesDataset], MultiHorizonTransform):
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

    @property
    @override
    def is_fitted(self) -> bool:
        return True  # Stateless transform by default, always considered fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        pass

    @override
    def fit_horizons(self, data: MultiHorizonTimeSeriesDataset) -> None:
        # Because the imputation computes a global statistic, we fit on the concatenated data
        flat_data = concat_horizon_datasets_rowwise(data)
        return self.fit(flat_data)

    @override
    def transform_horizons(self, data: MultiHorizonTimeSeriesDataset) -> MultiHorizonTimeSeriesDataset:
        transformed_data: MultiHorizonTimeSeriesDataset = {}
        for horizon, dataset in data.items():
            transformed_data[horizon] = self.transform(dataset)

        return transformed_data


__all__ = [
    "MultiHorizonTransform",
    "TimeSeriesTransform",
    "concat_horizon_datasets_rowwise",
]
