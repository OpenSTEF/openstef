# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Simple constant median forecasting models for educational and baseline purposes.

Provides basic forecasting models that predict constant values based on historical
medians. These models serve as educational examples and performance baselines.
"""

from typing import ClassVar, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import Any, LeadTime, Quantile
from openstef_models.explainability.mixins import ContributionsMixin, ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster


class ConstantMedianForecasterHyperParams(HyperParams):
    """Hyperparameter configuration for constant median forecaster."""

    constant: float = Field(
        default=0.01,
        description="Constant to add to the forecasts.",
    )


class ConstantMedianForecaster(Forecaster, ExplainableForecaster, ContributionsMixin):
    """Constant median-based forecaster for single horizon predictions.

    Predicts constant values based on historical quantiles from training data.
    Useful as a baseline model and for educational purposes.

    The forecaster computes quantile values during training and returns these
    constant values for all future predictions. Performance is typically poor
    but provides a simple baseline for comparison with more sophisticated models.

    Example:
        >>> from openstef_core.types import LeadTime, Quantile
        >>> from datetime import timedelta
        >>> forecaster = ConstantMedianForecaster(
        ...     quantiles=[Quantile(0.5), Quantile(0.1), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=ConstantMedianForecasterHyperParams(),
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

    hyperparams: ConstantMedianForecasterHyperParams = Field(
        default_factory=ConstantMedianForecasterHyperParams,
    )

    _quantile_values: dict[Quantile, float] = PrivateAttr(default_factory=dict[Quantile, float])

    @property
    @override
    def hparams(self) -> ConstantMedianForecasterHyperParams:
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
