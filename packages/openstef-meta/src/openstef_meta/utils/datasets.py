# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Ensemble Forecast Dataset.

Validated dataset for ensemble forecasters first stage output.
Implements methods to select quantile-specific ForecastInputDatasets for final learners.
Also supports constructing classifation targets based on pinball loss.
"""

from datetime import datetime, timedelta
from typing import Self, override

import pandas as pd

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.types import Quantile
from openstef_meta.utils.pinball_errors import calculate_pinball_errors

DEFAULT_TARGET_COLUMN = {Quantile(0.5): "load"}


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
            raise ValueError("Data columns do not match the expected number based on base learners and quantiles.")

    @property
    def target_series(self) -> pd.Series | None:
        """Return the target series if available."""
        if self.target_column in self.data.columns:
            return self.data[self.target_column]
        return None

    @staticmethod
    def get_learner_and_quantile(feature_names: pd.Index) -> tuple[list[str], list[Quantile]]:
        """Extract base learner names and quantiles from feature names.

        Args:
            feature_names: Index of feature names in the dataset.

        Returns:
            Tuple containing a list of base learner names and a list of quantiles.

        Raises:
            ValueError: If an invalid base learner name is found in a feature name.
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
        """Generate the feature name for a given base learner and quantile.

        Args:
            feature_name: Feature name string in the format "BaseLearner_Quantile".

        Returns:
            Tuple containing the base learner name and Quantile object.
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
            Series with categorical indicators of best-performing base learners.
        """

        # Calculate pinball loss for each base learner
        def column_pinball_losses(preds: pd.Series) -> pd.Series:
            return calculate_pinball_errors(y_true=target, y_pred=preds, quantile=quantile)

        pinball_losses = data.apply(column_pinball_losses)

        return pinball_losses.idxmin(axis=1)

    def select_quantile_classification(self, quantile: Quantile) -> ForecastInputDataset:
        """Select classification target for a specific quantile.

        Args:
            quantile: Quantile to select.

        Returns:
            Series containing binary indicators of best-performing base learners for the specified quantile.

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

    def select_quantile(self, quantile: Quantile) -> ForecastInputDataset:
        """Select data for a specific quantile.

        Args:
            quantile: Quantile to select.

        Returns:
            ForecastInputDataset containing base predictions for the specified quantile.
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
