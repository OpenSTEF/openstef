# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core meta model interfaces and configurations.

Provides the fundamental building blocks for implementing meta models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different meta model types
while ensuring full compatability with regular Forecasters.
"""

import logging
from typing import cast, override

import pandas as pd

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_meta.framework.base_learner import (
    BaseLearner,
    BaseLearnerHyperParams,
    BaseLearnerNames,
)
from openstef_meta.framework.final_learner import ForecastCombiner
from openstef_meta.utils.datasets import EnsembleForecastDataset
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
    _forecast_combiner: ForecastCombiner

    @property
    @override
    def is_fitted(self) -> bool:
        return all(x.is_fitted for x in self._base_learners) and self._forecast_combiner.is_fitted

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

        if self._forecast_combiner.has_features:
            self._forecast_combiner.final_learner_processing.fit(full_dataset)
            features = self._forecast_combiner.calculate_features(data=full_dataset)
        else:
            features = None

        sample_weights = None
        if data.sample_weight_column in data.data.columns:
            sample_weights = data.data.loc[:, data.sample_weight_column]

        self._forecast_combiner.fit(
            data=base_predictions,
            data_val=None,  # TODO ADD validation dataset support
            additional_features=features,
            sample_weights=sample_weights,
        )

        self._is_fitted = True

    def _predict_base_learners(self, data: ForecastInputDataset) -> EnsembleForecastDataset:
        """Generate predictions from base learners.

        Args:
            data: Input data for prediction.

        Returns:
            DataFrame containing base learner predictions.
        """
        base_predictions: dict[BaseLearnerNames, ForecastDataset] = {}
        for learner in self._base_learners:
            preds = learner.predict(data=data)
            name = cast(BaseLearnerNames, learner.__class__.__name__)
            base_predictions[name] = preds

        return EnsembleForecastDataset.from_forecast_datasets(base_predictions, target_series=data.target_series)

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

        if self._forecast_combiner.has_features:
            additional_features = self._forecast_combiner.calculate_features(data=data)
        else:
            additional_features = None

        return self._forecast_combiner.predict(
            data=base_predictions,
            additional_features=additional_features,
        )


__all__ = [
    "BaseLearner",
    "BaseLearnerHyperParams",
    "MetaForecaster",
]
