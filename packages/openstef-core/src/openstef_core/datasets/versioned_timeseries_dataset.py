# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Versioned time series dataset for efficient multi-part composition."""

import functools
import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal, Self, cast, override

import pandas as pd
from pydantic import FilePath

from openstef_core.datasets.mixins import DatasetMixin, TimeSeriesMixin
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validation import validate_disjoint_columns, validate_same_sample_intervals
from openstef_core.exceptions import TimeSeriesValidationError
from openstef_core.types import AvailableAt, LeadTime
from openstef_core.utils import timedelta_from_isoformat
from openstef_core.utils.pandas import combine_timeseries_indexes, unsafe_sorted_range_slice_idxs

type ConcatMode = Literal["left", "outer", "inner"]


class VersionedTimeSeriesDataset(TimeSeriesMixin, DatasetMixin):
    """A versioned time series dataset composed of multiple data parts.

    This class combines multiple TimeSeriesDataset instances into a unified
    dataset that tracks data availability over time. It provides methods to filter
    datasets by time ranges, availability constraints, and lead times, as well as
    select specific versions of the data for point-in-time reconstruction.

    The dataset is particularly useful for realistic backtesting scenarios where
    data arrives with delays or gets revised over time.

    Key motivation: This architecture solves the O(n²) space complexity problem
    that occurs when concatenating DataFrames with misaligned (timestamp, available_at)
    pairs. Instead of immediately combining data, it uses lazy composition that
    delays actual DataFrame concatenation until select_version() is called.

    Attributes:
        data_parts: List of TimeSeriesDataset instances that compose this dataset.

    Example:
        Create a versioned dataset by combining multiple data parts:

        >>> import pandas as pd
        >>> from datetime import datetime, timedelta
        >>>
        >>> # Create weather data part
        >>> weather_data = pd.DataFrame({
        ...     'temperature': [20.5],
        ...     'available_at': [datetime(2025, 1, 1, 16, 0)]
        ... }, index=pd.DatetimeIndex([datetime(2025, 1, 1, 10, 0)]))
        >>> weather_part = TimeSeriesDataset(weather_data, timedelta(hours=1))
        >>>
        >>> # Combine into versioned dataset
        >>> dataset = VersionedTimeSeriesDataset([weather_part])
        >>> dataset.is_versioned
        True

    Note:
        All data parts must have identical sample intervals and disjoint feature sets.
        The final dataset index is the union of all part indices, enabling flexible
        composition of data sources with different coverage periods.
    """

    data_parts: list[TimeSeriesDataset]

    _index: pd.DatetimeIndex
    _sample_interval: timedelta
    _feature_names: list[str]

    def __init__(
        self,
        data_parts: list[TimeSeriesDataset],
        *,
        index: pd.DatetimeIndex | None = None,
    ) -> None:
        """Initialize a versioned time series dataset from multiple parts.

        Args:
            data_parts: List of TimeSeriesDataset instances to combine. Must have
                identical sample intervals and disjoint feature sets.
            index: Optional explicit index for the combined dataset. If not provided,
                the union of all part indices will be used.

        Raises:
            TimeSeriesValidationError: If no data parts provided or validation fails.
        """
        if not data_parts:
            raise TimeSeriesValidationError("At least one data part must be provided.")

        if not all(part.is_versioned for part in data_parts):
            raise TimeSeriesValidationError("All data parts must be versioned datasets.")

        self._sample_interval = validate_same_sample_intervals(datasets=data_parts)
        self._feature_names = validate_disjoint_columns(datasets=data_parts)
        self._index = (
            index if index is not None else combine_timeseries_indexes(indexes=[part.index for part in data_parts])
        )
        self.data_parts = data_parts

    @property
    @override
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    @override
    def sample_interval(self) -> timedelta:
        return self._sample_interval

    @property
    @override
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    @override
    def is_versioned(self) -> bool:
        return True

    def _copy_with_data(self, data_parts: list[TimeSeriesDataset], index: pd.DatetimeIndex | None = None) -> Self:
        # Fast way to copy self with new data and skipping validation since invariants are preserved.
        new_instance = object.__new__(self.__class__)
        new_instance.__dict__.update(self.__dict__)
        new_instance.data_parts = data_parts
        new_instance._index = index if index is not None else self._index  # noqa: SLF001
        return new_instance

    @override
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        if start is None and end is None:
            return self

        start_idx, end_idx = unsafe_sorted_range_slice_idxs(data=cast(pd.Series, self.index), start=start, end=end)
        index = self.index[start_idx:end_idx]

        data_parts = [part.filter_by_range(start, end) for part in self.data_parts]
        return self._copy_with_data(data_parts=data_parts, index=index)

    @override
    def filter_by_available_before(self, available_before: datetime) -> Self:
        data_parts = [part.filter_by_available_before(available_before) for part in self.data_parts]
        return self._copy_with_data(data_parts=data_parts)

    @override
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        data_parts = [part.filter_by_available_at(available_at) for part in self.data_parts]
        return self._copy_with_data(data_parts=data_parts)

    @override
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        data_parts = [part.filter_by_lead_time(lead_time) for part in self.data_parts]
        return self._copy_with_data(data_parts=data_parts)

    @override
    def select_version(self) -> TimeSeriesDataset:
        selected_parts = [part.select_version().data for part in self.data_parts]
        combined_data = pd.concat(selected_parts, axis=1).reindex(self.index)
        return TimeSeriesDataset(data=combined_data, sample_interval=self.sample_interval)

    @override
    def to_parquet(self, path: FilePath) -> None:
        parts_df = [part.to_pandas() for part in self.data_parts]
        parts_metadata = [{**part_df.attrs, "columns": part_df.columns.tolist()} for part_df in parts_df]
        combined_data = pd.concat([part.data.assign(part_id=i) for i, part in enumerate(self.data_parts)], axis=0)
        combined_data.attrs["parts"] = json.dumps({"parts": parts_metadata})
        combined_data.to_parquet(path=path)

    @override
    @classmethod
    def read_parquet(
        cls,
        path: FilePath,
        *,
        sample_interval: timedelta | None = None,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
        horizon_column: str = "horizon",
    ) -> Self:
        df = pd.read_parquet(path=path)  # type: ignore
        if "parts" in df.attrs:
            parts_metadata = json.loads(df.attrs.get("parts", "{}")).get("parts", [])
            if len(parts_metadata) == 0:
                raise TimeSeriesValidationError("No data parts found in the parquet file.")

            parts: list[TimeSeriesDataset] = [
                TimeSeriesDataset(
                    data=df.loc[df.part_id == i, part_info["columns"]],  # type: ignore
                    sample_interval=timedelta_from_isoformat(part_info.get("sample_interval", "PT1H")),
                )
                for i, part_info in enumerate(parts_metadata)
            ]
        else:
            part = TimeSeriesDataset.read_parquet(
                path=path,
                sample_interval=sample_interval,
                timestamp_column=timestamp_column,
                available_at_column=available_at_column,
                horizon_column=horizon_column,
            )
            if not part.is_versioned and not part._version_column != part.available_at_column:  # noqa: SLF001
                raise TimeSeriesValidationError(
                    "Parquet file does not contain versioned data. Use TimeSeriesDataset.read_parquet() instead."
                )
            parts = [part]

        return cls(data_parts=parts)

    @classmethod
    def concat(cls, datasets: Sequence[Self], mode: ConcatMode) -> Self:
        """Concatenate multiple versioned datasets into a single dataset.

        Combines multiple VersionedTimeSeriesDataset instances using the specified
        concatenation mode. Supports different strategies for handling overlapping
        time indices across datasets.

        This method is useful when you have data from different sources or time
        periods that need to be combined while preserving their versioning
        information. For example, combining weather data from different providers
        or merging historical data with recent updates.

        Args:
            datasets: Sequence of VersionedTimeSeriesDataset instances to concatenate.
                Must contain at least one dataset.
            mode: Concatenation mode determining how to handle overlapping indices:
                - "left": Use indices from the first dataset only
                - "outer": Union of all indices across datasets
                - "inner": Intersection of all indices across datasets

        Returns:
            New VersionedTimeSeriesDataset containing all data parts from input datasets.

        Raises:
            TimeSeriesValidationError: If no datasets are provided for concatenation.
        """
        if not datasets:
            raise TimeSeriesValidationError("At least one dataset must be provided for concatenation.")

        data_parts = [part for dataset in datasets for part in dataset.data_parts]
        if mode == "outer" or len(datasets) == 1:
            return cls(data_parts=data_parts)

        if mode == "left":
            index = datasets[0].index
        elif mode == "inner":
            index = functools.reduce(lambda x, y: x.intersection(y), [part.index.unique() for part in data_parts])

        return cls(
            data_parts=[
                TimeSeriesDataset(data=part.data.loc[part.index.isin(index)])  # pyright: ignore[reportUnknownMemberType]
                for part in data_parts
            ],
            index=index,
        )

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        sample_interval: timedelta,
        *,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
    ) -> Self:
        """Create a VersionedTimeSeriesDataset from a single DataFrame.

        Convenience constructor for creating a versioned dataset from a single
        DataFrame containing all features.

        Args:
            data: DataFrame containing versioned time series data with timestamp
                and available_at columns.
            sample_interval: The regular interval between consecutive data points.
            available_at_column: Name of the column indicating when data became available.
                Default is 'available_at'.
            timestamp_column: Name of the column indicating the timestamps of the data.
                Default is 'timestamp'.

        Returns:
            New VersionedTimeSeriesDataset instance containing the data.

        Example:
            Create dataset from a single DataFrame:

            >>> import pandas as pd
            >>> from datetime import datetime, timedelta
            >>> data = pd.DataFrame({
            ...     'available_at': [datetime.fromisoformat('2025-01-01T10:05:00'),
            ...                      datetime.fromisoformat('2025-01-01T10:20:00')],
            ...     'load': [100.0, 120.0],
            ...     'temperature': [20.0, 22.0]
            ... }, index=pd.DatetimeIndex([datetime.fromisoformat('2025-01-01T10:00:00'),
            ...                            datetime.fromisoformat('2025-01-01T10:15:00')], name='timestamp'))
            >>> dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(minutes=15))
            >>> sorted(dataset.feature_names)
            ['load', 'temperature']

        Note:
            This is equivalent to creating a TimeSeriesDataset and then
            wrapping it in a VersionedTimeSeriesDataset, but more convenient
            for simple cases.
        """
        if not isinstance(data.index, pd.DatetimeIndex) and timestamp_column in data.columns:
            # Backwards compatibility: datasets with explicit timestamp column
            data = data.set_index(timestamp_column)

        return cls(
            data_parts=[
                TimeSeriesDataset(
                    data=data,
                    sample_interval=sample_interval,
                    available_at_column=available_at_column,
                )
            ]
        )

    def to_horizons(self, horizons: list[LeadTime]) -> TimeSeriesDataset:
        """Convert versioned dataset to horizon-based format for multiple lead times.

        Selects data for each specified horizon, adds a horizon column, and combines
        into a single TimeSeriesDataset. Useful for creating multi-horizon training data.

        Returns:
            TimeSeriesDataset with horizon column indicating forecast lead time.
        """
        horizon_dfs = [
            self.filter_by_lead_time(lead_time=horizon).select_version().data.assign(horizon=horizon.value)
            for horizon in horizons
        ]
        return TimeSeriesDataset(
            data=pd.concat(objs=horizon_dfs, axis=0),
            sample_interval=self.sample_interval,
        )
