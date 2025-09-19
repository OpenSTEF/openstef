# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level forecasting model that orchestrates the complete prediction pipeline.

Combines feature engineering, forecasting, and postprocessing into a unified interface.
Handles both single-horizon and multi-horizon forecasters while providing consistent
data transformation and validation.
"""

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.transforms import TransformPipeline
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ConfigurationError, NotFittedError, UnreachableStateError
from openstef_core.types import LeadTime
from openstef_models.models.forecasting import BaseForecaster, BaseHorizonForecaster
from openstef_models.transforms import FeatureEngineeringPipeline


class ForecastingModel(BaseModel):
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
    forecaster: BaseForecaster | BaseHorizonForecaster = Field(
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

    def _prepare_input_data(
        self,
        dataset: VersionedTimeSeriesDataset | TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> dict[LeadTime, ForecastInputDataset]:
        input_data = self.preprocessing.transform(data=dataset)
        return {
            lead_time: ForecastInputDataset.from_timeseries_dataset(
                dataset=timeseries_dataset,
                target_column=self.target_column,
                forecast_start=forecast_start,
            )
            for lead_time, timeseries_dataset in input_data.items()
        }

    @property
    def is_fitted(self) -> bool:
        """Check if the underlying forecaster has been trained.

        Returns:
            True if the forecaster is fitted and ready for prediction.
        """
        return self.forecaster.is_fitted

    def fit(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        """Train the forecasting model on the provided dataset.

        Fits the preprocessing pipeline and underlying forecaster. Handles both
        single-horizon and multi-horizon forecasters appropriately.

        Args:
            dataset: Historical time series data with features and target values.

        Raises:
            UnreachableStateError: If no data is available for horizon forecasting,
                indicating a violated invariant in the preprocessing pipeline.
        """
        # Fit the feature engineering transforms
        self.preprocessing.fit(data=dataset)

        # Transform the input data to a valid forecast input
        input_data = self._prepare_input_data(dataset=dataset)

        # Fit the model
        if isinstance(self.forecaster, BaseForecaster):
            self.forecaster.fit(input_data=input_data)
        else:
            horizon_input_data = next(iter(input_data.values()), None)
            if horizon_input_data is None:
                raise UnreachableStateError("No data available for horizon forecasting.")

            self.forecaster.fit_horizon(input_data=horizon_input_data)

    def predict(
        self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        """Generate forecasts using the trained model.

        Transforms input data through the preprocessing pipeline, generates predictions
        using the underlying forecaster, and applies postprocessing transformations.

        Args:
            dataset: Input time series data for generating forecasts.
            forecast_start: Starting time for forecasts. If None, uses dataset end time.

        Returns:
            Processed forecast dataset with predictions and uncertainty estimates.

        Raises:
            NotFittedError: If the model hasn't been trained yet.
            UnreachableStateError: If no data is available for horizon forecasting,
                indicating a violated invariant in the preprocessing pipeline.
        """
        if not self.is_fitted:
            raise NotFittedError(type(self.forecaster).__name__)

        # Transform the input data to a valid forecast input
        input_data = self._prepare_input_data(dataset=dataset, forecast_start=forecast_start)

        # Generate predictions
        if isinstance(self.forecaster, BaseForecaster):
            raw_forecasts = self.forecaster.predict(input_data=input_data)
        else:
            horizon_input_data = next(iter(input_data.values()), None)
            if horizon_input_data is None:
                raise UnreachableStateError("No data available for horizon forecasting.")

            raw_forecasts = self.forecaster.predict_horizon(input_data=horizon_input_data)

        return self.postprocessing.transform(raw_forecasts)
