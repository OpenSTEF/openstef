# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import cast, override

import pandas as pd

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError, TimeSeriesValidationError
from openstef_core.types import EnergyComponentType, Quantile


class ForecastInputDataset(TimeSeriesDataset):
    @override
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        target_column: str = "load",
        forecast_start: datetime | None = None,
    ) -> None:
        super().__init__(data, sample_interval)

        if target_column not in self.feature_names:
            raise MissingColumnsError(missing_columns=[target_column])

        self.target_column = target_column
        self.forecast_start = forecast_start

    def target_series(self) -> pd.Series:
        """Returns the target series from the dataset."""
        return self.data[self.target_column]


class ForecastDataset(TimeSeriesDataset):
    @override
    def __init__(self, data: pd.DataFrame, sample_interval: timedelta, forecast_start: datetime | None = None) -> None:
        super().__init__(data, sample_interval)

        if not all(Quantile.is_valid_quantile_string(col) for col in self.feature_names):
            raise TimeSeriesValidationError("All feature names must be valid quantile strings.")

        self._quantiles = [Quantile.parse(col) for col in self.feature_names]
        self.forecast_start = forecast_start or cast("pd.Series[pd.Timestamp]", self.index).min().to_pydatetime()

    @property
    def quantiles(self) -> list[Quantile]:
        """List of quantiles represented in the dataset's feature names."""
        return self._quantiles


class TimeSeriesEnergyComponentDataset(TimeSeriesDataset):
    @override
    def __init__(self, data: pd.DataFrame, sample_interval: timedelta) -> None:
        super().__init__(data, sample_interval)

        missing_columns = {item.value for item in EnergyComponentType} - set(self.feature_names)
        if missing_columns:
            raise MissingColumnsError(missing_columns=list(missing_columns))
