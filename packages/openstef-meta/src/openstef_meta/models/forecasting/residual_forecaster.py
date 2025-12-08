# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Hybrid Forecaster (Stacked LightGBM + Linear Model Gradient Boosting).

Provides method that attempts to combine the advantages of a linear model (Extraplolation)
and tree-based model (Non-linear patterns). This is acieved by training two base learners,
followed by a small linear model that regresses on the baselearners' predictions.
The implementation is based on sklearn's ResidualRegressor.
"""

import logging
from typing import override

import pandas as pd
from pydantic import Field, model_validator

from openstef_core.datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import (
    NotFittedError,
)
from openstef_core.mixins import HyperParams
from openstef_core.types import Quantile
from openstef_models.models.forecasting.forecaster import (
    Forecaster,
    ForecasterConfig,
)
from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster, LGBMHyperParams
from openstef_models.models.forecasting.lgbmlinear_forecaster import LGBMLinearForecaster, LGBMLinearHyperParams
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster, XGBoostHyperParams

logger = logging.getLogger(__name__)

ResidualBaseForecaster = LGBMForecaster | LGBMLinearForecaster | XGBoostForecaster | GBLinearForecaster
ResidualBaseForecasterHyperParams = LGBMHyperParams | LGBMLinearHyperParams | XGBoostHyperParams | GBLinearHyperParams


class ResidualHyperParams(HyperParams):
    """Hyperparameters for Stacked LGBM GBLinear Regressor."""

    primary_hyperparams: ResidualBaseForecasterHyperParams = Field(
        default=GBLinearHyperParams(),
        description="Primary model hyperparams. Defaults to GBLinearHyperParams.",
    )

    secondary_hyperparams: ResidualBaseForecasterHyperParams = Field(
        default=LGBMHyperParams(),
        description="Hyperparameters for the final learner. Defaults to LGBMHyperparams.",
    )

    primary_name: str = Field(
        default="primary_model",
        description="Name identifier for the primary model.",
    )

    secondary_name: str = Field(
        default="secondary_model",
        description="Name identifier for the secondary model.",
    )

    @model_validator(mode="after")
    def validate_names(self) -> "ResidualHyperParams":
        """Validate that primary and secondary names are not the same.

        Raises:
            ValueError: If primary and secondary names are the same.

        Returns:
            ResidualHyperParams: The validated hyperparameters.
        """
        if self.primary_name == self.secondary_name:
            raise ValueError("Primary and secondary model names must be different.")
        return self


class ResidualForecasterConfig(ForecasterConfig):
    """Configuration for Hybrid-based forecasting models."""

    hyperparams: ResidualHyperParams = ResidualHyperParams()

    verbosity: bool = Field(
        default=True,
        description="Enable verbose output from the Hybrid model (True/False).",
    )


class ResidualForecaster(Forecaster):
    """MetaForecaster that implements residual modeling.

    It takes in a primary forecaster and a residual forecaster. The primary forecaster makes initial predictions,
    and the residual forecaster models the residuals (errors) of the primary forecaster to improve overall accuracy.
    """

    Config = ResidualForecasterConfig
    HyperParams = ResidualHyperParams

    def __init__(self, config: ResidualForecasterConfig) -> None:
        """Initialize the Hybrid forecaster."""
        self._config = config

        self._primary_model: ResidualBaseForecaster = self._init_base_learners(
            config=config, base_hyperparams=[config.hyperparams.primary_hyperparams]
        )[0]

        self._secondary_model: list[ResidualBaseForecaster] = self._init_secondary_model(
            hyperparams=config.hyperparams.secondary_hyperparams
        )
        self.primary_name = config.hyperparams.primary_name
        self.secondary_name = config.hyperparams.secondary_name
        self._is_fitted = False

    def _init_secondary_model(self, hyperparams: ResidualBaseForecasterHyperParams) -> list[ResidualBaseForecaster]:
        """Initialize secondary model for residual forecasting.

        Returns:
            list[Forecaster]: List containing the initialized secondary model forecaster.
        """
        models: list[ResidualBaseForecaster] = []
        # Different datasets per quantile, so we need a model per quantile
        for q in self.config.quantiles:
            config = self._config.model_copy(update={"quantiles": [q]})
            secondary_model = self._init_base_learners(config=config, base_hyperparams=[hyperparams])[0]
            models.append(secondary_model)

        return models

    @staticmethod
    def _init_base_learners(
        config: ForecasterConfig, base_hyperparams: list[ResidualBaseForecasterHyperParams]
    ) -> list[ResidualBaseForecaster]:
        """Initialize base learners based on provided hyperparameters.

        Returns:
            list[Forecaster]: List of initialized base learner forecasters.
        """
        base_learners: list[ResidualBaseForecaster] = []
        horizons = config.horizons
        quantiles = config.quantiles

        for hyperparams in base_hyperparams:
            forecaster_cls = hyperparams.forecaster_class()
            config = forecaster_cls.Config(horizons=horizons, quantiles=quantiles)
            if "hyperparams" in forecaster_cls.Config.model_fields:
                config = config.model_copy(update={"hyperparams": hyperparams})

            base_learners.append(config.forecaster_from_config())

        return base_learners

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the Hybrid model to the training data.

        Args:
            data: Training data in the expected ForecastInputDataset format.
            data_val: Validation data for tuning the model (optional, not used in this implementation).

        """
        # Fit primary model
        self._primary_model.fit(data=data, data_val=data_val)

        # Reset forecast start date to ensure we fit on the full training set
        full_dataset = ForecastInputDataset(
            data=data.data,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
            forecast_start=data.index[0],
        )

        secondary_input = self._prepare_secondary_input(
            quantiles=self.config.quantiles,
            base_predictions=self._primary_model.predict(data=full_dataset),
            data=data,
        )
        # Predict primary model on validation data if provided
        if data_val is not None:
            full_val_dataset = ForecastInputDataset(
                data=data_val.data,
                sample_interval=data_val.sample_interval,
                target_column=data_val.target_column,
                forecast_start=data_val.index[0],
            )

            secondary_val_input = self._prepare_secondary_input(
                quantiles=self.config.quantiles,
                base_predictions=self._primary_model.predict(data=full_val_dataset),
                data=data_val,
            )
            # Fit secondary model on residuals
            [
                self._secondary_model[i].fit(data=secondary_input[q], data_val=secondary_val_input[q])
                for i, q in enumerate(secondary_input)
            ]

        else:
            # Fit secondary model on residuals
            [
                self._secondary_model[i].fit(data=secondary_input[q], data_val=None)
                for i, q in enumerate(secondary_input)
            ]

        self._is_fitted = True

    @property
    @override
    def is_fitted(self) -> bool:
        """Check the ResidualForecaster is fitted."""
        return self._is_fitted

    @staticmethod
    def _prepare_secondary_input(
        quantiles: list[Quantile],
        base_predictions: ForecastDataset,
        data: ForecastInputDataset,
    ) -> dict[Quantile, ForecastInputDataset]:
        """Adjust target series to be residuals for secondary model training.

        Args:
            quantiles: List of quantiles to prepare data for.
            base_predictions: Predictions from the primary model.
            data: Original input data.

        Returns:
            dict[Quantile, ForecastInputDataset]: Prepared datasets for each quantile.
        """
        predictions_quantiles: dict[Quantile, ForecastInputDataset] = {}
        sample_interval = data.sample_interval
        for q in quantiles:
            predictions = base_predictions.data[q.format()]
            df = data.data.copy()
            df[data.target_column] = data.target_series - predictions
            predictions_quantiles[q] = ForecastInputDataset(
                data=df,
                sample_interval=sample_interval,
                target_column=data.target_column,
                forecast_start=df.index[0],
            )

        return predictions_quantiles

    def _predict_secodary_model(self, data: ForecastInputDataset) -> ForecastDataset:
        predictions: dict[str, pd.Series] = {}
        for model in self._secondary_model:
            pred = model.predict(data=data)
            q = model.config.quantiles[0].format()
            predictions[q] = pred.data[q]

        return ForecastDataset(
            data=pd.DataFrame(predictions),
            sample_interval=data.sample_interval,
        )

    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Generate predictions using the ResidualForecaster model.

        Args:
            data: Input data for prediction.

        Returns:
            ForecastDataset containing the predictions.

        Raises:
            NotFittedError: If the ResidualForecaster instance is not fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError("The ResidualForecaster instance is not fitted yet. Call 'fit' first.")

        primary_predictions = self._primary_model.predict(data=data).data

        secondary_predictions = self._predict_secodary_model(data=data).data

        final_predictions = primary_predictions + secondary_predictions

        return ForecastDataset(
            data=final_predictions,
            sample_interval=data.sample_interval,
        )

    def predict_contributions(self, data: ForecastInputDataset, *, scale: bool = True) -> pd.DataFrame:
        """Generate prediction contributions using the ResidualForecaster model.

        Args:
            data: Input data for prediction contributions.
            scale: Whether to scale contributions to sum to 1. Defaults to True.

        Returns:
            pd.DataFrame containing the prediction contributions.
        """
        primary_predictions = self._primary_model.predict(data=data).data

        secondary_predictions = self._predict_secodary_model(data=data).data

        if not scale:
            primary_contributions = primary_predictions
            primary_name = self._primary_model.__class__.__name__
            primary_contributions.columns = [f"{primary_name}_{q}" for q in primary_contributions.columns]

            secondary_contributions = secondary_predictions
            secondary_name = self._secondary_model[0].__class__.__name__
            secondary_contributions.columns = [f"{secondary_name}_{q}" for q in secondary_contributions.columns]

            return pd.concat([primary_contributions, secondary_contributions], axis=1)

        primary_contributions = primary_predictions.abs() / (primary_predictions.abs() + secondary_predictions.abs())
        primary_contributions.columns = [f"{self.primary_name}_{q}" for q in primary_contributions.columns]

        secondary_contributions = secondary_predictions.abs() / (
            primary_predictions.abs() + secondary_predictions.abs()
        )
        secondary_contributions.columns = [f"{self.secondary_name}_{q}" for q in secondary_contributions.columns]

        return pd.concat([primary_contributions, secondary_contributions], axis=1)

    @property
    def config(self) -> ResidualForecasterConfig:
        """Get the configuration of the ResidualForecaster.

        Returns:
            ResidualForecasterConfig: The configuration of the forecaster.
        """
        return self._config


__all__ = ["ResidualForecaster", "ResidualForecasterConfig", "ResidualHyperParams"]
