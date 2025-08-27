import functools
import logging
import operator
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import Self, Sequence, cast, overload, override

import pandas as pd
from pydantic import FilePath

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import InvalidColumnTypeError, MissingColumnsError, TimeSeriesValidationError
from openstef_core.types import AvailableAt, LeadTime
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat

_logger = logging.getLogger(__name__)


class VerionedTimeSeriesMixin(ABC):
    @abstractmethod
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self: ...

    @abstractmethod
    def filter_by_available_at(self, available_at: AvailableAt) -> Self: ...

    @abstractmethod
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self: ...

    @abstractmethod
    def select_version(self, available_before: datetime | None) -> TimeSeriesDataset: ...


class VersionedTimeSeriesPart(VerionedTimeSeriesMixin):
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        index: pd.DatetimeIndex | None = None,
        timestamp_column: str = "timestamp",
        available_at_column: str = "available_at",
    ) -> None:
        # Validate required columns
        missing_columns = {timestamp_column, available_at_column} - set(data.columns)
        if missing_columns:
            raise MissingColumnsError(missing_columns=list(missing_columns))

        # Validate timestamp and available_at columns types
        _validate_datetime_column(data[timestamp_column], timestamp_column)
        _validate_datetime_column(data[available_at_column], available_at_column)

        # Ensure invariant: data is at all times sorted by (timestamp, available_at) asc.
        if not data.attrs.get("is_sorted", False):
            data = data.sort_values(by=[timestamp_column, available_at_column], ascending=[True, True])
            data.attrs["is_sorted"] = True

        self.data = data
        self.sample_interval = sample_interval
        self.timestamp_column = timestamp_column
        self.available_at_column = available_at_column
        self.index = index or cast(pd.DatetimeIndex, pd.DatetimeIndex(self.data[timestamp_column]))
        self.feature_names = list(set(self.data.columns) - {self.timestamp_column, self.available_at_column})

    @override
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        start_idx, end_idx = _sorted_range_slice_idxs(data=self.data[self.timestamp_column], start=start, end=end)
        data_filtered = self.data.iloc[start_idx:end_idx]
        return self._copy_with_data(data_filtered)

    @override
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        cutoff = self.data[self.timestamp_column].dt.floor("D") - pd.Timedelta(available_at.lag_from_day)
        data_filtered = self.data[self.data[self.available_at_column] <= cutoff]
        return self._copy_with_data(data_filtered, reindex=False)

    @override
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        cutoff = self.data[self.timestamp_column] - pd.Timedelta(lead_time.value)
        data_filtered = self.data[self.data[self.available_at_column] <= cutoff]
        return self._copy_with_data(data_filtered, reindex=False)

    @override
    def select_version(self, available_before: datetime | None) -> TimeSeriesDataset:
        if available_before is not None:
            data = self.data[self.data[self.available_at_column] <= available_before]
        else:
            data = self.data

        return TimeSeriesDataset(
            data=(
                data.drop_duplicates(subset=self.timestamp_column, keep="first")
                .drop(columns=[self.available_at_column])
                .set_index(self.timestamp_column)
            ),
            sample_interval=self.sample_interval,
        )

    def _copy_with_data(self, new_data: pd.DataFrame, reindex: bool = True) -> Self:
        return self.__class__(
            data=new_data,
            sample_interval=self.sample_interval,
            index=self.index if not reindex else None,
            timestamp_column=self.timestamp_column,
            available_at_column=self.available_at_column,
        )

    def to_parquet(self, path: FilePath) -> None:
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        self.data.attrs["timestamp_column"] = self.timestamp_column
        self.data.attrs["available_at_column"] = self.available_at_column
        self.data.attrs["is_sorted"] = True
        self.data.to_parquet(path)

    @classmethod
    def read_parquet(cls, path: FilePath) -> Self:
        data = pd.read_parquet(path)
        if "sample_interval" not in data.attrs:
            _logger.warning(
                "Parquet file does not contain 'sample_interval' attribute. Using default value of 15 minutes."
            )

        sample_interval = timedelta_from_isoformat(data.attrs.get("sample_interval", "PT15M"))
        timestamp_column = data.attrs.get("timestamp_column", "timestamp")
        available_at_column = data.attrs.get("available_at_column", "available_at")

        return cls(
            data=data,
            sample_interval=sample_interval,
            timestamp_column=timestamp_column,
            available_at_column=available_at_column,
        )


class VersionedTimeSeriesDataset(VerionedTimeSeriesMixin):
    index: pd.DatetimeIndex
    sample_interval: timedelta
    feature_names: list[str]

    def __init__(
        self,
        data_parts: list[VersionedTimeSeriesPart],
        index: pd.DatetimeIndex,
    ) -> None:
        if not data_parts:
            raise TimeSeriesValidationError("At least one data part must be provided.")

        _validate_same_sample_intervals(data_parts)
        _validate_disjoint_columns(data_parts)

        self.data_parts = data_parts
        self.sample_interval = data_parts[0].sample_interval
        self.index = index
        self.feature_names = functools.reduce(operator.iadd, [part.feature_names for part in data_parts], [])

    @override
    def filter_by_range(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        start_idx, end_idx = _sorted_range_slice_idxs(
            data=cast(pd.Series[pd.Timestamp], self.index), start=start, end=end
        )
        index = self.index[start_idx:end_idx]

        return self.__class__(
            data_parts=[part.filter_by_range(start, end) for part in self.data_parts],
            index=index,
        )

    @override
    def filter_by_available_at(self, available_at: AvailableAt) -> Self:
        return self.__class__(
            data_parts=[part.filter_by_available_at(available_at) for part in self.data_parts],
            index=self.index,
        )

    @override
    def filter_by_lead_time(self, lead_time: LeadTime) -> Self:
        return self.__class__(
            data_parts=[part.filter_by_lead_time(lead_time) for part in self.data_parts],
            index=self.index,
        )

    @override
    def select_version(self, available_before: datetime | None) -> TimeSeriesDataset:
        selected_parts = [part.select_version(available_before).data for part in self.data_parts]
        combined_data = pd.concat(selected_parts, axis=1).loc[self.index]
        return TimeSeriesDataset(data=combined_data, sample_interval=self.sample_interval)

    def to_parquet(self, path: Path) -> None: ...

    @classmethod
    def read_parquet(cls, path: Path) -> Self: ...


def _validate_disjoint_columns(dfs: Sequence[VersionedTimeSeriesPart]) -> None:
    all_features: list[str] = functools.reduce(operator.iadd, [d.feature_names for d in dfs], [])
    if len(all_features) != len(set(all_features)):
        duplicate_features = [item for item, count in Counter(all_features).items() if count > 1]
        raise TimeSeriesValidationError("Datasets have overlapping feature names: " + ", ".join(duplicate_features))


def _validate_datetime_column(series: pd.Series, column_name: str) -> None:
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise InvalidColumnTypeError(column_name, expected_type="datetime", actual_type=str(series.dtype))


def _validate_same_sample_intervals(datasets: Sequence[VersionedTimeSeriesPart]) -> None:
    sample_intervals = {d.sample_interval for d in datasets}
    if len(sample_intervals) > 1:
        raise TimeSeriesValidationError(
            "Datasets have different sample intervals: " + ", ".join(map(str, sample_intervals))
        )


def _sorted_range_slice_idxs(
    data: pd.Series[pd.Timestamp], start: datetime | None, end: datetime | None
) -> tuple[int, int]:
    start_idx = data.searchsorted(start, side="left") if start else 0
    end_idx = data.searchsorted(end, side="left") if end else len(data)
    return start_idx, end_idx
