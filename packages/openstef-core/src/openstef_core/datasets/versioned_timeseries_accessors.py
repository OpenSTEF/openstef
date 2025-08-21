# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import operator
from datetime import datetime, timedelta
from functools import reduce
from typing import Literal, cast

import pandas as pd

from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.datasets.validation import check_features_are_disjoint, check_sample_intervals

type ConcatMode = Literal["left", "outer", "inner"]


class ConcatenatedVersionedTimeSeries(VersionedTimeSeriesMixin):
    def __init__(self, datasets: list[VersionedTimeSeriesMixin], mode: ConcatMode) -> None:
        if len(datasets) < 2:  # noqa: PLR2004
            msg = "At least two datasets are required for ConcatFeaturewise."
            raise ValueError(msg)

        check_features_are_disjoint(datasets)
        check_sample_intervals(datasets)
        self._datasets = datasets
        self._features = reduce(operator.iadd, [d.feature_names for d in datasets], [])
        self._mode = mode

        indexes = [d.index for d in datasets]
        match mode:
            case "left":
                self._index = indexes[0]
            case "outer":
                self._index = reduce(lambda x, y: cast(pd.DatetimeIndex, x.union(y)), indexes)
            case "inner":
                self._index = reduce(lambda x, y: x.intersection(y), indexes)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    def feature_names(self) -> list[str]:
        return self._features

    @property
    def sample_interval(self) -> timedelta:
        return self._datasets[0].sample_interval

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        dataframes = [d.get_window(start=start, end=end, available_before=available_before) for d in self._datasets]
        return pd.concat(dataframes, axis=1)


class RestrictedHorizonVersionedTimeSeries(VersionedTimeSeriesMixin):
    def __init__(self, dataset: VersionedTimeSeriesMixin, horizon: datetime) -> None:
        self._dataset = dataset
        self._horizon = horizon

    @property
    def feature_names(self) -> list[str]:
        return self._dataset.feature_names

    @property
    def sample_interval(self) -> timedelta:
        return self._dataset.sample_interval

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._dataset.index

    @property
    def horizon(self) -> datetime:
        return self._horizon

    def get_window(self, start: datetime, end: datetime, available_before: datetime | None = None) -> pd.DataFrame:
        if available_before is not None and available_before > self._horizon:
            msg = f"Available before {available_before} is greater than the horizon."
            raise ValueError(msg)

        return self._dataset.get_window(start=start, end=end, available_before=available_before or self._horizon)


class VersionedTimeSeriesAccessors:
    @staticmethod
    def concat_featurewise(datasets: list[VersionedTimeSeriesMixin], mode: ConcatMode) -> VersionedTimeSeriesMixin:
        if len(datasets) == 1:
            return datasets[0]

        return ConcatenatedVersionedTimeSeries(datasets=datasets, mode=mode)

    @staticmethod
    def restrict_horizon(dataset: VersionedTimeSeriesMixin, horizon: datetime) -> VersionedTimeSeriesMixin:
        return RestrictedHorizonVersionedTimeSeries(dataset, horizon)
