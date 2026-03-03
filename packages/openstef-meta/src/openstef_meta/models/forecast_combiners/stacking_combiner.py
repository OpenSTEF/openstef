# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Stacking Forecast Combiner.

A meta-regressor per quantile is trained on top of the base forecasters' predictions.
Each quantile gets its own stacking model (e.g., GBLinear or LGBM).
"""

import logging
from functools import partial
from typing import override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.datasets.validated_datasets import EnsembleForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins.predictor import HyperParams
from openstef_core.types import Quantile
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner
from openstef_meta.utils.datasets import combine_forecast_input_datasets
from openstef_models.explainability.mixins import ContributionsMixin, ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster

logger = logging.getLogger(__name__)


class StackingCombiner(ForecastCombiner):
    """Stacking combiner: one meta-regressor per quantile on top of base forecaster outputs.

    Accepts a template ``meta_forecaster`` (a fully-configured :class:`Forecaster`
    instance).  During initialisation the template is cloned once per quantile —
    each clone receives a single quantile while horizons are taken from the
    combiner's own configuration.
    """

    meta_forecaster: Forecaster = Field(
        exclude=True,
        description="Template forecaster cloned per quantile as the stacking meta-forecaster.",
    )

    _is_fitted: bool = PrivateAttr(default=False)
    _models: dict[Quantile, Forecaster] = PrivateAttr(default_factory=dict[Quantile, Forecaster])

    @property
    @override
    def hparams(self) -> HyperParams:
        return self.meta_forecaster.hparams

    def model_post_init(self, _context: object, /) -> None:
        """Clone the template forecaster once per quantile."""
        models: dict[Quantile, Forecaster] = {}
        for q in self.quantiles:
            models[q] = self.meta_forecaster.model_copy(
                update={"quantiles": [q], "horizons": [self.max_horizon]},
            )
        self._models = models

    @staticmethod
    def _prepare_input(
        data: EnsembleForecastDataset,
        quantile: Quantile,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastInputDataset:
        input_data = data.get_base_predictions_for_quantile(quantile=quantile)
        if additional_features is not None:
            input_data = combine_forecast_input_datasets(input_data=input_data, additional_features=additional_features)
        return input_data

    @property
    @override
    def is_fitted(self) -> bool:
        return all(x.is_fitted for x in self._models.values())

    @override
    def fit(
        self,
        data: EnsembleForecastDataset,
        data_val: EnsembleForecastDataset | None = None,
        additional_features: ForecastInputDataset | None = None,
    ) -> None:
        for q in self.quantiles:
            input_data = self._prepare_input(data, q, additional_features)

            target_dropna = partial(pd.DataFrame.dropna, subset=[input_data.target_column])  # pyright: ignore[reportUnknownMemberType]
            input_data = input_data.pipe_pandas(target_dropna)

            self._models[q].fit(data=input_data, data_val=None)

    @override
    def predict(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        predictions = [
            self._models[q].predict(data=self._prepare_input(data, q, additional_features)).data for q in self.quantiles
        ]
        return ForecastDataset(data=pd.concat(predictions, axis=1), sample_interval=data.sample_interval)

    @override
    def predict_contributions(
        self,
        data: EnsembleForecastDataset,
        additional_features: ForecastInputDataset | None = None,
    ) -> TimeSeriesDataset:
        frames: list[pd.DataFrame] = []
        for q in self.quantiles:
            model = self._models[q]
            if not isinstance(model, ContributionsMixin):
                msg = f"Model {type(model).__name__} does not support predict_contributions."
                raise NotImplementedError(msg)
            frames.append(model.predict_contributions(data=self._prepare_input(data, q, additional_features)).data)

        contributions = pd.concat(frames, axis=1)
        target_series = data.target_series
        if target_series is not None:
            contributions[data.target_column] = target_series
        return TimeSeriesDataset(data=contributions, sample_interval=data.sample_interval)

    @property
    @override
    def feature_importances(self) -> pd.DataFrame:
        frames = [m.feature_importances for m in self._models.values() if isinstance(m, ExplainableForecaster)]
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


__all__ = ["StackingCombiner"]
