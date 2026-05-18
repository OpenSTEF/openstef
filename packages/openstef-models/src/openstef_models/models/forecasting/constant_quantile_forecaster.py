# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Simple constant quantile forecasting model.

Provides basic forecasting models that predict constant values based on historical
quantiles. These models can be used as a simple baseline or naive fallback model.
"""

from typing import ClassVar, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.exceptions import InputValidationError, NotFittedError
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import Any, LeadTime, Quantile
from openstef_models.explainability.mixins import ContributionsMixin, ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster


class ConstantQuantileForecasterHyperParams(HyperParams):
    """Hyperparameter configuration for constant quantile forecaster."""

    constant: float = Field(
        default=0.0,
        description="Constant to add to the forecasts.",
    )


class ConstantQuantileForecaster(Forecaster, ExplainableForecaster, ContributionsMixin):
    """Constant quantile-based forecaster for single horizon predictions.

    Predicts constant values based on historical quantiles from training data.

    The forecaster computes quantile values during training and returns these
    constant values for all future predictions. Performance is typically poor
    but provides a simple baseline for comparison with more sophisticated models.

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> forecaster = ConstantQuantileForecaster(
        ...     quantiles=[Quantile(0.5), Quantile(0.1), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=ConstantQuantileForecasterHyperParams(),
        ... )
        >>> forecaster.fit(training_data)  # doctest: +SKIP
        >>> predictions = forecaster.predict(test_data)  # doctest: +SKIP
    """

    _VERSION: ClassVar[int] = 2

    horizons: list[LeadTime] = Field(
        default=...,
        description=(
            "Lead times for predictions, accounting for data availability and versioning cutoffs. "
            "Each horizon defines how far ahead the model should predict."
        ),
        min_length=1,
        max_length=1,
    )

    hyperparams: ConstantQuantileForecasterHyperParams = Field(
        default_factory=ConstantQuantileForecasterHyperParams,
    )

    _quantile_values: dict[Quantile, float] = PrivateAttr(default_factory=dict[Quantile, float])

    @property
    @override
    def hparams(self) -> ConstantQuantileForecasterHyperParams:
        return self.hyperparams

    @override
    @classmethod
    def _migrate_state(cls, state: dict[str, Any], from_version: int, to_version: int) -> dict[str, Any]:
        if from_version <= 1:
            state["quantile_values"] = state.pop("quantile_values_v1", {})
        return state

    @property
    @override
    def is_fitted(self) -> bool:
        return len(self._quantile_values) > 0

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        if data.target_series.isna().all():
            raise InputValidationError("Training data must contain at least one non-NaN value in the target column.")
        self._quantile_values = {quantile: data.target_series.quantile(quantile) for quantile in self.quantiles}

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        forecast_index = data.create_forecast_range(horizon=self.max_horizon)
        return ForecastDataset(
            data=pd.DataFrame(
                data={
                    quantile.format(): self._quantile_values[quantile] + self.hyperparams.constant
                    for quantile in self.quantiles
                },
                index=forecast_index,
            ),
            sample_interval=data.sample_interval,
            target_column=data.target_column,
        )

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[1.0],
            index=["load"],
            columns=[quantile.format() for quantile in self.quantiles],
        )

    @override
    def predict_contributions(self, data: ForecastInputDataset) -> TimeSeriesDataset:
        """Return uniform contributions."""
        input_data = data.input_data(start=data.forecast_start)
        contribs_df = pd.DataFrame(1.0, index=input_data.index, columns=["bias"])
        return TimeSeriesDataset(data=contribs_df, sample_interval=data.sample_interval)
