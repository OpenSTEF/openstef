# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Validated dataset classes for time series forecasting.

Specialized dataset classes with domain-specific validation for different stages
of the forecasting pipeline. These datasets inherit from TimeSeriesDataset and add
validation to catch data quality issues early.
"""

from datetime import datetime, timedelta
from typing import Self, override

import numpy as np
import numpy.typing as npt
import pandas as pd

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.exceptions import MissingColumnsError
from openstef_core.types import EnergyComponentType, LeadTime, Quantile


class ForecastInputDataset(TimeSeriesDataset):
    """Time series dataset for forecasting with validated target column.

    Used for training and prediction data where a specific target column
    must exist. The target column represents the value being forecasted.

    Invariants:
        - Target column must exist in the dataset
        - Inherits all TimeSeriesDataset guarantees (sorted timestamps, consistent intervals)

    Attrs:
        target_column: Name of the target column to forecast.
        sample_weight_column: Name of the column containing sample weights.
        forecast_start: Optional timestamp indicating when the forecast period starts.

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
        >>> len(dataset.target_series)
        3
        >>> len(dataset.sample_weight_series)
        3

    See Also:
        TimeSeriesDataset: Base class for time series datasets.
        ForecastDataset: For storing probabilistic forecast results.
        TimeSeriesEnergyComponentDataset: For energy component analysis.
    """

    target_column: str
    sample_weight_column: str

    _forecast_start: datetime | None

    @override
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta = timedelta(minutes=15),
        forecast_start: datetime | None = None,
        *,
        horizon_column: str = "horizon",
        available_at_column: str = "available_at",
        check_frequency: bool = False,
        sample_weight_column: str = "sample_weight",
        target_column: str = "load",
    ) -> None:
        self.target_column = data.attrs.get("target_column", target_column)
        self.sample_weight_column = data.attrs.get("sample_weight_column", sample_weight_column)
        if "forecast_start" in data.attrs:
            self._forecast_start = datetime.fromisoformat(data.attrs["forecast_start"])
        else:
            self._forecast_start = forecast_start

        validate_required_columns(data, required_columns=[self.target_column])

        super().__init__(
            data=data,
            sample_interval=sample_interval,
            horizon_column=horizon_column,
            available_at_column=available_at_column,
            check_frequency=check_frequency,
        )
        self._internal_columns.add(self.sample_weight_column)
        self._feature_names = [col for col in self.data.columns if col not in self._internal_columns]

    @property
    def forecast_start(self) -> datetime:
        """Get the forecast start timestamp.

        Returns:
            Datetime indicating when the forecast period starts.
        """
        return self._forecast_start if self._forecast_start is not None else self.data.index.min().to_pydatetime()

    @property
    def target_series(self) -> pd.Series:
        """Extract the target time series from the dataset.

        Returns:
            Time series containing target values with original datetime index.
        """
        return self.data[self.target_column]

    @property
    def sample_weight_series(self) -> pd.Series:
        """Extract the sample weight time series from the dataset, if it exists.

        Returns:
            Time series containing sample weights with original datetime index,
            or None if the sample weight column does not exist.
        """
        if self.sample_weight_column in self.data.columns:
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
        drop_columns = [self.target_column, self.sample_weight_column]
        if self._version_column is not None:
            drop_columns.append(self._version_column)

        input_data: pd.DataFrame = self.data.drop(columns=drop_columns, errors="ignore")
        if start is not None:
            input_data = input_data[input_data.index >= pd.Timestamp(start)]

        return input_data

    @classmethod
    def from_timeseries(
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

    def create_forecast_range(self, horizon: LeadTime) -> pd.DatetimeIndex:
        """Create forecast index for given horizon starting from forecast_start.

        Args:
            horizon: Lead time horizon for the forecast.

        Returns:
            DatetimeIndex representing the forecast timestamps.
        """
        return pd.date_range(
            start=self.forecast_start,
            end=self.forecast_start + horizon.value,
            freq=self.sample_interval,
            name="timestamp",
        )

    @override
    def to_pandas(self) -> pd.DataFrame:
        df = super().to_pandas()
        df.attrs["target_column"] = self.target_column
        df.attrs["sample_weight_column"] = self.sample_weight_column
        df.attrs["forecast_start"] = self.forecast_start.isoformat()
        return df


class ForecastDataset(TimeSeriesDataset):
    """Time series dataset containing probabilistic forecasts with quantile estimates.

    Contains forecast results with column names following quantile naming convention
    (e.g., 'quantile_P50' for median). Enables consistent handling of probabilistic
    forecasts with uncertainty quantification.

    Invariants:
        - All columns must be valid quantile strings (e.g., 'quantile_P10')
        - Inherits all TimeSeriesDataset guarantees (sorted timestamps, consistent intervals)

    Attrs:
        forecast_start: Timestamp indicating when the forecast period starts.
        quantiles: List of Quantile values represented in the dataset.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from datetime import timedelta
        >>> forecast_data = pd.DataFrame({
        ...     'load': [100, np.nan],
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

    forecast_start: datetime
    quantiles: list[Quantile]
    target_column: str

    @override
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta = timedelta(minutes=15),
        forecast_start: datetime | None = None,
        target_column: str = "load",
        *,
        horizon_column: str = "horizon",
        available_at_column: str = "available_at",
        standard_deviation_column: str = "stdev",
    ) -> None:
        if "forecast_start" in data.attrs:
            self.forecast_start = datetime.fromisoformat(data.attrs["forecast_start"])
        else:
            self.forecast_start = forecast_start if forecast_start is not None else data.index.min().to_pydatetime()
        self.target_column = data.attrs.get("target_column", target_column)
        self.standard_deviation_column = data.attrs.get("standard_deviation_column", standard_deviation_column)

        super().__init__(
            data=data,
            sample_interval=sample_interval,
            horizon_column=horizon_column,
            available_at_column=available_at_column,
        )

        exclude_columns = {target_column, standard_deviation_column}
        quantile_feature_names = [col for col in self.feature_names if col not in exclude_columns]
        if not all(Quantile.is_valid_quantile_string(col) for col in quantile_feature_names):
            raise ValueError("All feature names must be valid quantile strings.")

        self.quantiles = [Quantile.parse(col) for col in quantile_feature_names]

    @property
    def target_series(self) -> pd.Series | None:
        """Extract the target time series from the dataset.

        Returns:
            Time series containing target values with original datetime index.
        """
        if self.target_column not in self.data.columns:
            return None
        return self.data[self.target_column]

    @property
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

    @property
    def standard_deviation_series(self) -> pd.Series:
        """Extract the standard deviation series if it exists.

        Returns:
            Time series containing standard deviation values with original datetime index.

        Raises:
            MissingColumnsError: If the standard deviation column is not found.
        """
        if self.standard_deviation_column not in self.data.columns:
            raise MissingColumnsError(missing_columns=[self.standard_deviation_column])
        return self.data[self.standard_deviation_column]  # pyright: ignore[reportUnknownVariableType]

    @property
    def quantiles_data(self) -> pd.DataFrame:
        """Extract DataFrame containing only the quantile forecast columns.

        Returns:
            DataFrame with quantile columns and original datetime index.
        """
        quantile_columns = [q.format() for q in self.quantiles]
        return self.data[quantile_columns]

    def filter_quantiles(self, quantiles: list[Quantile]) -> Self:
        """Select a subset of quantiles from the forecast dataset.

        Args:
            quantiles: List of Quantile values to select.

        Returns:
            New ForecastDataset containing only the specified quantile columns.
        """
        selected_quantiles = [q.format() for q in quantiles]
        validate_required_columns(self.data, required_columns=selected_quantiles)

        all_quantiles = [q.format() for q in self.quantiles]
        drop_columns = list(set(all_quantiles) - set(selected_quantiles))
        data_filtered = self.data.drop(columns=drop_columns)

        result = self._copy_with_data(data=data_filtered)
        result.quantiles = quantiles
        return result

    @override
    def to_pandas(self) -> pd.DataFrame:
        df = super().to_pandas()
        df.attrs["target_column"] = self.target_column
        df.attrs["forecast_start"] = self.forecast_start.isoformat()
        df.attrs["standard_deviation_column"] = self.standard_deviation_column
        return df

    @classmethod
    def from_timeseries(
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
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta = timedelta(minutes=15),
        *,
        horizon_column: str = "horizon",
        available_at_column: str = "available_at",
    ) -> None:
        validate_required_columns(
            data,
            required_columns=[item.value for item in EnergyComponentType],
        )
        super().__init__(
            data=data,
            sample_interval=sample_interval,
            horizon_column=horizon_column,
            available_at_column=available_at_column,
        )


class EnsembleForecastDataset(TimeSeriesDataset):
    """First stage output format for ensemble forecasters."""

    forecast_start: datetime
    quantiles: list[Quantile]
    forecaster_names: list[str]
    target_column: str

    @override
    def __init__(
        self,
        data: pd.DataFrame,
        sample_interval: timedelta = timedelta(minutes=15),
        forecast_start: datetime | None = None,
        target_column: str = "load",
        *,
        horizon_column: str = "horizon",
        available_at_column: str = "available_at",
    ) -> None:
        if "forecast_start" in data.attrs:
            self.forecast_start = datetime.fromisoformat(data.attrs["forecast_start"])
        else:
            self.forecast_start = forecast_start if forecast_start is not None else data.index.min().to_pydatetime()
        self.target_column = data.attrs.get("target_column", target_column)

        super().__init__(
            data=data,
            sample_interval=sample_interval,
            horizon_column=horizon_column,
            available_at_column=available_at_column,
        )
        quantile_feature_names = [col for col in self.feature_names if col != target_column]

        self.forecaster_names, self.quantiles = self.get_learner_and_quantile(pd.Index(quantile_feature_names))
        n_cols = len(self.forecaster_names) * len(self.quantiles)
        if len(data.columns) not in {n_cols + 1, n_cols}:
            raise ValueError("Data columns do not match the expected number based on base forecasters and quantiles.")

    @property
    def target_series(self) -> pd.Series | None:
        """Return the target series if available."""
        if self.target_column in self.data.columns:
            return self.data[self.target_column]
        return None

    @staticmethod
    def get_learner_and_quantile(feature_names: pd.Index) -> tuple[list[str], list[Quantile]]:
        """Extract base forecaster names and quantiles from feature names.

        Args:
            feature_names: Index of feature names in the dataset.

        Returns:
            Tuple containing a list of base forecaster names and a list of quantiles.

        Raises:
            ValueError: If an invalid base forecaster name is found in a feature name.
        """
        forecasters: set[str] = set()
        quantiles: set[Quantile] = set()

        for feature_name in feature_names:
            quantile_part = "_".join(feature_name.split("_")[-2:])
            learner_part = feature_name[: -(len(quantile_part) + 1)]
            if not Quantile.is_valid_quantile_string(quantile_part):
                msg = f"Column has no valid quantile string: {feature_name}"
                raise ValueError(msg)

            forecasters.add(learner_part)
            quantiles.add(Quantile.parse(quantile_part))

        return list(forecasters), list(quantiles)

    @staticmethod
    def get_quantile_feature_name(feature_name: str) -> tuple[str, Quantile]:
        """Generate the feature name for a given base forecaster and quantile.

        Args:
            feature_name: Feature name string in the format "model_Quantile".

        Returns:
            Tuple containing the base forecaster name and Quantile object.
        """
        learner_part, quantile_part = feature_name.split("_", maxsplit=1)
        return learner_part, Quantile.parse(quantile_part)

    @classmethod
    def from_forecast_datasets(
        cls,
        datasets: dict[str, ForecastDataset],
        target_series: pd.Series | None = None,
        sample_weights: pd.Series | None = None,
    ) -> Self:
        """Create an EnsembleForecastDataset from multiple ForecastDatasets.

        Args:
            datasets: Dict of ForecastDatasets to combine.
            target_series: Optional target series to include in the dataset.
            sample_weights: Optional sample weights series to include in the dataset.

        Returns:
            EnsembleForecastDataset combining all input datasets.
        """
        ds1 = next(iter(datasets.values()))
        additional_columns: dict[str, pd.Series] = {}
        if isinstance(ds1.target_series, pd.Series):
            additional_columns[ds1.target_column] = ds1.target_series
        elif target_series is not None:
            additional_columns[ds1.target_column] = target_series

        sample_weight_column = "sample_weight"
        if sample_weights is not None:
            additional_columns[sample_weight_column] = sample_weights

        combined_data = pd.DataFrame({
            f"{learner}_{q.format()}": ds.data[q.format()] for learner, ds in datasets.items() for q in ds.quantiles
        }).assign(**additional_columns)

        return cls(
            data=combined_data,
            sample_interval=ds1.sample_interval,
            forecast_start=ds1.forecast_start,
            target_column=ds1.target_column,
        )

    @staticmethod
    def _prepare_classification(data: pd.DataFrame, target: pd.Series, quantile: Quantile) -> pd.Series:
        """Prepare data for classification tasks by converting quantile columns to binary indicators.

        Args:
            data: DataFrame containing quantile predictions.
            target: Series containing true target values.
            quantile: Quantile for which to prepare classification data.

        Returns:
            Series with categorical indicators of best-performing base forecasters.
        """
        y_true = np.asarray(target)

        def _column_losses(preds: pd.Series) -> npt.NDArray[np.floating]:
            y_pred = np.asarray(preds)
            errors = y_true - y_pred
            return np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)

        losses_per_forecaster = data.apply(_column_losses)

        return losses_per_forecaster.idxmin(axis=1)

    def get_best_forecaster_labels(self, quantile: Quantile) -> ForecastInputDataset:
        """Get labels indicating the best-performing base forecaster for each sample at a specific quantile.

        Creates a dataset where each sample's target is labeled with the name of the base forecaster
        that performed best, determined by pinball loss. Used as classification target for training
        the final learner.

        Args:
            quantile: Quantile to select.

        Returns:
            ForecastInputDataset where the target column contains labels of the best-performing
            base forecaster for each sample.

        Raises:
            ValueError: If the target column is not found in the dataset.
        """
        if self.target_column not in self.data.columns:
            msg = f"Target column '{self.target_column}' not found in dataset."
            raise ValueError(msg)

        selected_columns = [f"{learner}_{quantile.format()}" for learner in self.forecaster_names]
        prediction_data = self.data[selected_columns].copy()
        prediction_data.columns = self.forecaster_names

        target = self._prepare_classification(
            data=prediction_data,
            target=self.data[self.target_column],
            quantile=quantile,
        )
        prediction_data[self.target_column] = target
        return ForecastInputDataset(
            data=prediction_data,
            sample_interval=self.sample_interval,
            target_column=self.target_column,
            forecast_start=self.forecast_start,
        )

    def get_base_predictions_for_quantile(self, quantile: Quantile) -> ForecastInputDataset:
        """Get base forecaster predictions for a specific quantile.

        Args:
            quantile: Quantile to select.

        Returns:
            ForecastInputDataset containing predictions from all base forecasters at the specified quantile.
        """
        selected_columns = [f"{learner}_{quantile.format()}" for learner in self.forecaster_names]
        selected_columns.append(self.target_column)
        prediction_data = self.data[selected_columns].copy()
        prediction_data.columns = [*self.forecaster_names, self.target_column]

        return ForecastInputDataset(
            data=prediction_data,
            sample_interval=self.sample_interval,
            target_column=self.target_column,
            forecast_start=self.forecast_start,
        )


__all__ = [
    "EnergyComponentDataset",
    "EnsembleForecastDataset",
    "ForecastDataset",
    "ForecastInputDataset",
]
