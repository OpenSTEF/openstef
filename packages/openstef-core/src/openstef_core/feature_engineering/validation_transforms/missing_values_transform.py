# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from enum import Enum

import numpy as np
from sklearn.impute import SimpleImputer

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


class ImputationStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class MissingValuesTransformConfig(BaseConfig):
    """Configuration class for MissingValuesCheck transform.

    Note: fill_value is used when imputation_strategy is CONSTANT.
    """

    missing_value: float = np.nan
    imputation_strategy: ImputationStrategy
    fill_value: int | float | str | None = None
    no_fill_future_values_features: list[str] = []  # TODO: Use this...


class MissingValuesTransform(TimeSeriesTransform):
    """Transform that checks for, imputes and drops missing values in time series data."""

    def __init__(self, config: MissingValuesTransformConfig):
        self.config = config

        self.imputer_: SimpleImputer = SimpleImputer(
            strategy=self.config.imputation_strategy.value,
            fill_value=self.config.fill_value,
            missing_values=self.config.missing_value,
            keep_empty_features=False,
        )
        self.imputer_.set_output(transform="pandas")

    def fit(self, data: TimeSeriesDataset) -> None:
        self.imputer_ = self.imputer_.fit(data.data)

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        data.data = self.imputer_.transform(data.data)
        return data
