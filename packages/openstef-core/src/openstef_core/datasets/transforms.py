# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from abc import abstractmethod

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeseriesDataset


class TimeSeriesTransform:
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

        This method should apply a transformation to the input data and return a new instance of VersionedTimeseriesDataset.

        Args:
            data: The input versioned time series data to be transformed.

        Returns:
            A new instance of VersionedTimeseriesDataset containing the transformed data.
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
