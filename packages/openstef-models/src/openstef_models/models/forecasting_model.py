# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level forecasting model that orchestrates the complete prediction pipeline.

Combines feature engineering, forecasting, and postprocessing into a unified interface.
Handles both single-horizon and multi-horizon forecasters while providing consistent
data transformation and validation.
"""

from datetime import datetime
from typing import Any, Self, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from openstef_core.base_model import BaseModel
from openstef_core.datasets import (
    ForecastDataset,
    ForecastInputDataset,
    MultiHorizon,
    TimeSeriesDataset,
    VersionedTimeSeriesDataset,
)
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.mixins import Predictor, State, TransformPipeline
from openstef_core.types import LeadTime
from openstef_models.models.forecasting import Forecaster, HorizonForecaster
from openstef_models.transforms import FeatureEngineeringPipeline


class FitResult(BaseModel):
    input_dataset: VersionedTimeSeriesDataset | TimeSeriesDataset

    input_data_train: MultiHorizon[ForecastInputDataset]
    input_data_val: MultiHorizon[ForecastInputDataset] | None

    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Training metrics collected during the fit process.",
    )


class ForecastingModel(BaseModel, Predictor[VersionedTimeSeriesDataset | TimeSeriesDataset, ForecastDataset]):
    """Complete forecasting pipeline combining preprocessing, prediction, and postprocessing.

    Orchestrates the full forecasting workflow by managing feature engineering,
    model training/prediction, and result postprocessing. Automatically handles
    the differences between single-horizon and multi-horizon forecasters while
    ensuring data consistency and validation throughout the pipeline.

    Invariants:
        - fit() must be called before predict()
        - Forecaster and preprocessing horizons must match during initialization

    Example:
        Basic forecasting workflow:

        >>> from openstef_models.models.forecasting.constant_median_forecaster import (
        ...     ConstantMedianForecaster, ConstantMedianForecasterConfig
        ... )
        >>> from openstef_models.transforms import FeatureEngineeringPipeline
        >>>
        >>> # Note: This is a conceptual example showing the API structure
        >>> # Real usage requires implemented forecaster classes
        >>> forecaster = ConstantMedianForecaster(
        ...     config=ConstantMedianForecasterConfig(horizons=[LeadTime.from_string("PT36H")])
        ... )
        >>> preprocessing = FeatureEngineeringPipeline(horizons=forecaster.config.horizons)
        >>>
        >>> # Create and train model
        >>> model = ForecastingModel(
        ...     forecaster=forecaster,
        ...     preprocessing=preprocessing
        ... )
        >>> # model.fit(training_data)
        >>> #
        >>> # Generate forecasts
        >>> # forecasts = model.predict(new_data)
    """

    preprocessing: FeatureEngineeringPipeline = Field(
        default=...,
        description="Feature engineering pipeline for transforming raw input data into model-ready features.",
    )
    forecaster: Forecaster | HorizonForecaster = Field(
        default=...,
        description="Underlying forecasting algorithm, either single-horizon or multi-horizon.",
    )
    postprocessing: TransformPipeline[ForecastDataset] = Field(
        default_factory=TransformPipeline[ForecastDataset],
        description="Postprocessing pipeline for transforming model outputs into final forecasts.",
    )
    target_column: str = Field(
        default="load",
        description="Name of the target variable column in datasets.",
    )

    @model_validator(mode="after")
    def _validate_horizons_match(self) -> Self:
        if self.forecaster.config.horizons != self.preprocessing.horizons:
            message = (
                f"The forecaster horizons ({self.forecaster.config.horizons}) do not match the "
                f"preprocessing horizons ({self.preprocessing.horizons})."
            )
            raise ConfigurationError(message)

        return self

    def prepare_forecast_input(
        self,
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> MultiHorizon[ForecastInputDataset]:
        """Prepare input data for forecasting by applying preprocessing and ensuring target column consistency.

        Extracts the target series from the input data, applies the preprocessing pipeline,
        and creates ForecastInputDataset instances for each horizon with the target column
        properly assigned.

        Args:
            data: Input time series dataset containing features and target values.
            forecast_start: Starting time for forecasts. Passed to ForecastInputDataset.

        Returns:
            MultiHorizon dataset with ForecastInputDataset for each configured horizon.
        """
        if isinstance(data, VersionedTimeSeriesDataset):
            target_series = data.select_version().data[self.target_column]
        else:
            target_series = data.data[self.target_column]

        input_data = self.preprocessing.transform(data=data)

        return input_data.map_horizons(
            lambda dataset: ForecastInputDataset(
                # Reassign target column to ensure it exists and is unchanged
                data=dataset.data.assign(**{self.target_column: target_series}),
                sample_interval=data.sample_interval,
                target_column=self.target_column,
                forecast_start=forecast_start,
            )
        )

    @property
    @override
    def is_fitted(self) -> bool:
        return self.forecaster.is_fitted

    @override
    def fit(
        self,
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
        data_val: VersionedTimeSeriesDataset | TimeSeriesDataset | None = None,
    ) -> FitResult:
        """Train the forecasting model on the provided dataset.

        Fits the preprocessing pipeline and underlying forecaster. Handles both
        single-horizon and multi-horizon forecasters appropriately.

        Args:
            data: Historical time series data with features and target values.
            data_val: The validation data to evaluate and tune the predictor on (optional).

        Returns:
            FitResult containing training details and metrics.
        """
        # Fit the feature engineering transforms
        self.preprocessing.fit(data=data)

        # Transform the input data to a valid forecast input
        input_data_train = self.prepare_forecast_input(data=data)
        input_data_val = self.prepare_forecast_input(data=data_val) if data_val is not None else None

        # If validation dataset is not explicitly provided, apply a default split
        if input_data_val is None:
            pass

        # Fit the model
        prediction = self._fit_model(data=input_data_train, data_val=input_data_val)

        # Fit the postprocessing transforms
        self.postprocessing.fit(data=prediction)

        return FitResult(
            input_dataset=data,
            input_data_train=input_data_train,
            input_data_val=input_data_val,
            metrics={},  # TODO(#705): collect and return training metrics  # noqa: FIX002
        )

    def _fit_model(
        self,
        data: dict[LeadTime, ForecastInputDataset],
        data_val: dict[LeadTime, ForecastInputDataset] | None = None,
    ) -> ForecastDataset:
        if isinstance(self.forecaster, Forecaster):
            prediction = self.forecaster.fit_predict(data=data, data_val=data_val)
        else:
            horizon_input_data = data[self.preprocessing.horizons[0]]
            horizon_input_data_val = data_val[self.preprocessing.horizons[0]] if data_val else None
            prediction = self.forecaster.fit_predict(data=horizon_input_data, data_val=horizon_input_data_val)

        return prediction

    @override
    def predict(
        self, data: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        """Generate forecasts using the trained model.

        Transforms input data through the preprocessing pipeline, generates predictions
        using the underlying forecaster, and applies postprocessing transformations.

        Args:
            data: Input time series data for generating forecasts.
            forecast_start: Starting time for forecasts. If None, uses data end time.

        Returns:
            Processed forecast dataset with predictions and uncertainty estimates.

        Raises:
            NotFittedError: If the model hasn't been trained yet.
        """
        if not self.is_fitted:
            raise NotFittedError(type(self.forecaster).__name__)

        # Transform the input data to a valid forecast input
        input_data = self.prepare_forecast_input(data=data, forecast_start=forecast_start)

        # Empty out the target column in the to predict range
        for input_data_horizon in input_data.values():
            if forecast_start is None:
                input_data_horizon.data[self.target_column] = np.nan
            else:
                mask = input_data_horizon.index >= pd.Timestamp(forecast_start)
                input_data_horizon.data.loc[mask, self.target_column] = np.nan

        # Generate predictions
        if isinstance(self.forecaster, Forecaster):
            raw_forecasts = self.forecaster.predict(data=input_data)
        else:
            horizon_input_data = input_data[self.preprocessing.horizons[0]]
            raw_forecasts = self.forecaster.predict(data=horizon_input_data)

        return self.postprocessing.transform(raw_forecasts)

    @override
    def to_state(self) -> State:
        return {
            "target_column": self.target_column,
            "preprocessing": self.preprocessing.to_state(),
            "forecaster": self.forecaster.to_state(),
            "postprocessing": self.postprocessing.to_state(),
        }

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, Any], state)
        return self.__class__(
            target_column=state["target_column"],
            preprocessing=self.preprocessing.from_state(state["preprocessing"]),
            forecaster=self.forecaster.from_state(state["forecaster"]),
            postprocessing=self.postprocessing.from_state(state["postprocessing"]),
        )


__all__ = ["ForecastingModel"]
