# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from datetime import timedelta
from pathlib import Path
from typing import Self, cast

import pandas as pd

from openstef_core.datasets.mixins import TimeSeriesMixin
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat

_logger = logging.getLogger(__name__)


class TimeSeriesDataset(TimeSeriesMixin):
    data: pd.DataFrame
    _sample_interval: timedelta

    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
    ) -> None:
        super().__init__()

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data index must be a pandas DatetimeIndex.")

        if not data.attrs.get("is_sorted", False):
            data = data.sort_index(ascending=True)
            data.attrs["is_sorted"] = True

        self._sample_interval = sample_interval
        self.data = data

    @property
    def feature_names(self) -> list[str]:
        return self.data.columns.tolist()

    @property
    def sample_interval(self) -> timedelta:
        return self._sample_interval

    @property
    def index(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.data.index)

    def to_parquet(
        self,
        path: Path,
    ) -> None:
        self.data.attrs["sample_interval"] = timedelta_to_isoformat(self._sample_interval)
        self.data.attrs["is_sorted"] = True
        self.data.to_parquet(path)

    @classmethod
    def from_parquet(
        cls,
        path: Path,
    ) -> Self:
        """Create a TimeseriesDataset from a parquet file.

        This factory method loads data from a parquet file and initializes a TimeseriesDataset.

        Returns:
           TimeseriesDataset: A new TimeseriesDataset initialized with the data from the parquet file.
        """
        data = pd.read_parquet(path)
        if "sample_interval" not in data.attrs:
            _logger.warning(
                "Parquet file does not contain 'sample_interval' attribute. Using default value of 15 minutes."
            )

        sample_interval = timedelta_from_isoformat(data.attrs.get("sample_interval", "PT15M"))

        return cls(
            data=data,
            sample_interval=sample_interval,
        )


__all__ = ["TimeSeriesDataset"]
