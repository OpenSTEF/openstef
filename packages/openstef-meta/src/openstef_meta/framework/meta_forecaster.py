# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core meta model interfaces and configurations.

Provides the fundamental building blocks for implementing meta models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different meta model types
while ensuring full compatability with regular Forecasters.
"""

import logging
from typing import override

import pandas as pd

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.types import Quantile
from openstef_meta.framework.base_learner import (
    BaseLearner,
    BaseLearnerHyperParams,
)
from openstef_meta.framework.final_learner import FinalLearner
from openstef_models.models.forecasting.forecaster import (
    Forecaster,
    ForecasterConfig,
)

logger = logging.getLogger(__name__)


class MetaForecaster(Forecaster):
    """Abstract class for Meta forecasters combining multiple models."""

    _config: ForecasterConfig

    @staticmethod
    def _init_base_learners(
        config: ForecasterConfig, base_hyperparams: list[BaseLearnerHyperParams]
    ) -> list[BaseLearner]:
        """Initialize base learners based on provided hyperparameters.

        Returns:
            list[Forecaster]: List of initialized base learner forecasters.
        """
        base_learners: list[BaseLearner] = []
        horizons = config.horizons
        quantiles = config.quantiles

        for hyperparams in base_hyperparams:
            forecaster_cls = hyperparams.forecaster_class()
            config = forecaster_cls.Config(horizons=horizons, quantiles=quantiles)
            if "hyperparams" in forecaster_cls.Config.model_fields:
                config = config.model_copy(update={"hyperparams": hyperparams})

            base_learners.append(config.forecaster_from_config())

        return base_learners

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @property
    def feature_importances(self) -> pd.DataFrame:
        """Placeholder for feature importances across base learners and final learner."""
        raise NotImplementedError("Feature importances are not implemented for EnsembleForecaster.")
        # TODO(#745): Make MetaForecaster explainable

    @property
    def model_contributions(self) -> pd.DataFrame:
        """Placeholder for model contributions across base learners and final learner."""
        raise NotImplementedError("Model contributions are not implemented for EnsembleForecaster.")
        # TODO(#745): Make MetaForecaster explainable


class EnsembleForecaster(MetaForecaster):
    """Abstract class for Meta forecasters combining multiple base learners and a final learner."""

    _config: ForecasterConfig
    _base_learners: list[BaseLearner]
    _final_learner: FinalLearner

    @property
    @override
    def is_fitted(self) -> bool:
        return all(x.is_fitted for x in self._base_learners) and self._final_learner.is_fitted

    @property
    @override
    def config(self) -> ForecasterConfig:
        return self._config

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the Hybrid model to the training data.

        Args:
            data: Training data in the expected ForecastInputDataset format.
            data_val: Validation data for tuning the model (optional, not used in this implementation).

        """
        # Fit base learners
        [x.fit(data=data, data_val=data_val) for x in self._base_learners]

        # Reset forecast start date to ensure we predict on the full dataset
        full_dataset = ForecastInputDataset(
            data=data.data,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.index[0],
        )

        base_predictions = self._predict_base_learners(data=full_dataset)

        if self._final_learner.has_features:
            self._final_learner.final_learner_processing.fit(full_dataset)
            features = self._final_learner.calculate_features(data=full_dataset)
        else:
            features = None

        quantile_datasets = self._prepare_input_final_learner(
            base_predictions=base_predictions,
            quantiles=self._config.quantiles,
            target_series=data.target_series,
        )

        sample_weights = None
        if data.sample_weight_column in data.data.columns:
            sample_weights = data.data.loc[:, data.sample_weight_column]

        self._final_learner.fit(
            base_learner_predictions=quantile_datasets,
            additional_features=features,
            sample_weights=sample_weights,
        )

        self._is_fitted = True

    def _predict_base_learners(self, data: ForecastInputDataset) -> dict[type[BaseLearner], ForecastDataset]:
        """Generate predictions from base learners.

        Args:
            data: Input data for prediction.

        Returns:
            DataFrame containing base learner predictions.
        """
        base_predictions: dict[type[BaseLearner], ForecastDataset] = {}
        for learner in self._base_learners:
            preds = learner.predict(data=data)
            base_predictions[learner.__class__] = preds

        return base_predictions

    @staticmethod
    def _prepare_input_final_learner(
        quantiles: list[Quantile],
        base_predictions: dict[type[BaseLearner], ForecastDataset],
        target_series: pd.Series,
    ) -> dict[Quantile, ForecastInputDataset]:
        """Prepare input data for the final learner based on base learner predictions.

        Args:
            quantiles: List of quantiles to prepare data for.
            base_predictions: Predictions from base learners.
            target_series: Actual target series for reference.

        Returns:
            dictionary mapping Quantiles to ForecastInputDatasets.
        """
        predictions_quantiles: dict[Quantile, ForecastInputDataset] = {}
        sample_interval = base_predictions[next(iter(base_predictions))].sample_interval
        target_name = str(target_series.name)

        for q in quantiles:
            df = pd.DataFrame({
                learner.__name__: preds.data[Quantile(q).format()] for learner, preds in base_predictions.items()
            })
            df[target_name] = target_series

            predictions_quantiles[q] = ForecastInputDataset(
                data=df,
                sample_interval=sample_interval,
                target_column=target_name,
                forecast_start=df.index[0],
            )

        return predictions_quantiles

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        full_dataset = ForecastInputDataset(
            data=data.data,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.index[0],
        )

        base_predictions = self._predict_base_learners(data=full_dataset)

        final_learner_input = self._prepare_input_final_learner(
            quantiles=self._config.quantiles, base_predictions=base_predictions, target_series=data.target_series
        )

        if self._final_learner.has_features:
            additional_features = self._final_learner.calculate_features(data=data)
        else:
            additional_features = None

        return self._final_learner.predict(
            base_learner_predictions=final_learner_input, additional_features=additional_features
        )


__all__ = [
    "BaseLearner",
    "BaseLearnerHyperParams",
    "MetaForecaster",
]
