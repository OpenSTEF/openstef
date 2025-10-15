# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Validated dataset classes for time series forecasting.

Specialized dataset classes with domain-specific validation for different stages
of the forecasting pipeline. These datasets inherit from TimeSeriesDataset and add
validation to catch data quality issues early.
"""

from datetime import datetime, timedelta
from typing import Self, cast, override

import pandas as pd

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.exceptions import MissingColumnsError, TimeSeriesValidationError
from openstef_core.types import EnergyComponentType, Quantile


class ForecastInputDataset(TimeSeriesDataset):
    """Time series dataset for forecasting with validated target column.

    Used for training and prediction data where a specific target column
    must exist. The target column represents the value being forecasted.

    Invariants:
        - Target column must exist in the dataset
        - Inherits all TimeSeriesDataset guarantees (sorted timestamps, consistent intervals)

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110],
        ...     'temperature': [20, 22, 21],
        ...     'weights': [1.0, 0.5, 1.0],
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='h'))
        >>> dataset = ForecastInputDataset(
        ...     data=data,
        ...     sample_interval=timedelta(hours=1),
        ...     target_column='load',
        ...     sample_weight_column='weights',
        ... )
        >>> dataset.target_column
        'load'
        >>> dataset.sample_weight_column
        'weights'
        >>> len(dataset.target_series())
        3
        >>> len(dataset.sample_weight_series())
        3

    See Also:
        TimeSeriesDataset: Base class for time series datasets.
        ForecastDataset: For storing probabilistic forecast results.
        TimeSeriesEnergyComponentDataset: For energy component analysis.
    """

    @override
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        target_column: str | None = None,
        sample_weight_column: str = "sample_weight",
        forecast_start: datetime | None = None,
        *,
        is_sorted: bool = False,
    ) -> None:
        """Initialize dataset with target column validation.

        Args:
            data: Time series data with DatetimeIndex.
            sample_interval: Time interval between consecutive data points.
            target_column: Name of the target column to forecast.
            sample_weight_column: Name of column with sample weights.
            forecast_start: Optional timestamp indicating forecast start.
            is_sorted: Whether the data is already sorted by index.

        Raises:
            MissingColumnsError: If target_column is not found in data.
        """
        super().__init__(data=data, sample_interval=sample_interval, is_sorted=is_sorted)

        self.target_column: str = data.attrs.get("target_column", target_column) or "load"
        if self.target_column not in self.feature_names:
            raise MissingColumnsError(missing_columns=[self.target_column])

        self.sample_weight_column = sample_weight_column
        self.forecast_start = forecast_start
        self.data.attrs["target_column"] = self.target_column

    def target_series(self) -> pd.Series:
        """Extract the target time series from the dataset.

        Returns:
            Time series containing target values with original datetime index.
        """
        return self.data[self.target_column]

    def sample_weight_series(self) -> pd.Series:
        """Extract the sample weight time series from the dataset, if it exists.

        Returns:
            Time series containing sample weights with original datetime index,
            or None if the sample weight column does not exist.
        """
        if self.sample_weight_column in self.feature_names:
            return self.data[self.sample_weight_column]
        return pd.Series(1, index=self.index)

    def input_data(self, start: datetime | None = None) -> pd.DataFrame:
        """Extract the input features excluding the target column.

        Args:
            start: Optional datetime to filter data from. If provided, only includes
                data points with timestamps at or after this date.

        Returns:
            DataFrame containing input features with original datetime index.
        """
        input_data: pd.DataFrame = self.data.drop(
            columns=[self.target_column, self.sample_weight_column], errors="ignore"
        )
        if start is not None:
            input_data = input_data[input_data.index >= pd.Timestamp(start)]

        return input_data

    @classmethod
    def from_timeseries_dataset(
        cls,
        dataset: TimeSeriesDataset,
        target_column: str = "load",
        forecast_start: datetime | None = None,
    ) -> Self:
        """Create ForecastInputDataset from a generic TimeSeriesDataset.

        Args:
            dataset: Input TimeSeriesDataset to convert.
            target_column: Name of the target column to forecast.
            forecast_start: Optional timestamp indicating forecast start.

        Returns:
            Instance of ForecastInputDataset with specified target column.
        """
        return cls(
            data=dataset.data,
            sample_interval=dataset.sample_interval,
            target_column=target_column,
            forecast_start=forecast_start,
        )


class ForecastDataset(TimeSeriesDataset):
    """Time series dataset containing probabilistic forecasts with quantile estimates.

    Contains forecast results with column names following quantile naming convention
    (e.g., 'quantile_P50' for median). Enables consistent handling of probabilistic
    forecasts with uncertainty quantification.

    Invariants:
        - All columns must be valid quantile strings (e.g., 'quantile_P10')
        - Inherits all TimeSeriesDataset guarantees (sorted timestamps, consistent intervals)

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> forecast_data = pd.DataFrame({
        ...     'quantile_P10': [90, 95],
        ...     'quantile_P50': [100, 110],
        ...     'quantile_P90': [115, 125]
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='h'))
        >>> dataset = ForecastDataset(forecast_data, timedelta(hours=1))
        >>> len(dataset.quantiles)
        3
        >>> dataset.quantiles[1]
        0.5

    See Also:
        TimeSeriesDataset: Base class for time series datasets.
        ForecastInputDataset: For preparing forecasting input data.
        Quantile: Type for handling quantile values and naming conventions.
    """

    @override
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta,
        forecast_start: datetime | None = None,
        *,
        is_sorted: bool = False,
    ) -> None:
        """Initialize dataset with quantile column validation.

        Args:
            data: DataFrame with quantile columns (e.g., 'quantile_P50').
            sample_interval: Time interval between consecutive forecast points.
            forecast_start: Timestamp when forecast period begins. Defaults to
                minimum timestamp from data if not provided.
            is_sorted: Whether the data is already sorted by index.

        Raises:
            TimeSeriesValidationError: If any columns are not valid quantile strings.
        """
        super().__init__(data=data, sample_interval=sample_interval, is_sorted=is_sorted)

        if not all(Quantile.is_valid_quantile_string(col) for col in self.feature_names):
            raise TimeSeriesValidationError("All feature names must be valid quantile strings.")

        self._quantiles = [Quantile.parse(col) for col in self.feature_names]
        self.forecast_start = forecast_start or cast("pd.Series[pd.Timestamp]", self.index).min().to_pydatetime()

    @property
    def quantiles(self) -> list[Quantile]:
        """List of quantiles represented in the dataset.

        Returns:
            Quantile values (0-1 scale) corresponding to dataset columns.
        """
        return self._quantiles

    def median_series(self) -> pd.Series:
        """Extract the median (50th percentile) forecast series.

        Returns:
            Time series containing median forecast values with original datetime index.

        Raises:
            MissingColumnsError: If the median quantile column is not found.
        """
        median_col = Quantile(0.5).format()
        if median_col not in self.feature_names:
            raise MissingColumnsError(missing_columns=[median_col])
        return self.data[median_col]

    def select_quantiles(self, quantiles: list[Quantile]) -> Self:
        """Select a subset of quantiles from the forecast dataset.

        Args:
            quantiles: List of Quantile values to select.

        Returns:
            New ForecastDataset containing only the specified quantile columns.

        Raises:
            MissingColumnsError: If any requested quantile columns are not found.
        """
        quantile_columns = [q.format() for q in quantiles]
        missing_columns = set(quantile_columns) - set(self.feature_names)
        if missing_columns:
            raise MissingColumnsError(missing_columns=list(missing_columns))

        return self.__class__(
            data=self.data[quantile_columns],
            sample_interval=self.sample_interval,
            forecast_start=self.forecast_start,
            is_sorted=True,
        )


class EnergyComponentDataset(TimeSeriesDataset):
    """Time series dataset for energy generation by component type.

    Validates that all required energy component columns (wind, solar, other)
    are present. Used for energy sector analysis and component-specific forecasting.

    Invariants:
        - Must contain columns for all energy component types
        - Inherits all TimeSeriesDataset guarantees (sorted timestamps, consistent intervals)

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> energy_data = pd.DataFrame({
        ...     'wind': [50, 60],
        ...     'solar': [30, 40],
        ...     'other': [20, 25]
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='h'))
        >>> dataset = EnergyComponentDataset(energy_data, timedelta(hours=1))
        >>> 'wind' in dataset.feature_names
        True
        >>> len(dataset.feature_names)
        3

    See Also:
        TimeSeriesDataset: Base class for time series datasets.
        ForecastInputDataset: For general forecasting input data.
        EnergyComponentType: Enum defining required energy component types.
    """

    @override
    def __init__(self, data: pd.DataFrame, sample_interval: timedelta) -> None:
        """Initialize dataset with energy component validation.

        Args:
            data: DataFrame with columns for each energy component type.
            sample_interval: Time interval between consecutive data points.

        Raises:
            MissingColumnsError: If any required energy component columns are missing.
        """
        super().__init__(data, sample_interval)

        missing_columns = {item.value for item in EnergyComponentType} - set(self.feature_names)
        if missing_columns:
            raise MissingColumnsError(missing_columns=list(missing_columns))


__all__ = [
    "EnergyComponentDataset",
    "ForecastDataset",
    "ForecastInputDataset",
]
