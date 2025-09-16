# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import ConfigurationError, ModelNotFittedError
from openstef_core.types import LeadTime
from openstef_models.models.forecasting import BaseForecaster, BaseHorizonForecaster
from openstef_models.transforms import FeaturePipeline, PostprocessingPipeline


class ForecastingModel:
    preprocessing: FeaturePipeline
    forecaster: BaseForecaster | BaseHorizonForecaster
    postprocessing: PostprocessingPipeline
    target_column: str

    def __init__(
        self,
        forecaster: BaseForecaster | BaseHorizonForecaster,
        preprocessing: FeaturePipeline | None = None,
        postprocessing: PostprocessingPipeline | None = None,
        target_column: str = "load",
    ):
        if forecaster.config.horizons != self.preprocessing.horizons:
            message = (
                f"The forecaster horizons ({forecaster.config.horizons}) do not match the "
                "preprocessing horizons ({self.preprocessing.horizons})."
            )
            raise ConfigurationError(message)

        self.preprocessing = preprocessing or FeaturePipeline()
        self.forecaster = forecaster
        self.postprocessing = postprocessing or PostprocessingPipeline()
        self.target_column = target_column

    def _prepare_input_data(
        self,
        dataset: VersionedTimeSeriesDataset | TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> dict[LeadTime, ForecastInputDataset]:
        input_data = self.preprocessing.transform(dataset=dataset)
        return {
            key: ForecastInputDataset.from_timeseries_dataset(
                dataset=value,
                target_column=self.target_column,
                forecast_start=forecast_start,
            )
            for key, value in input_data.items()
        }

    @property
    def is_fitted(self) -> bool:
        return self.forecaster.is_fitted

    def fit(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        # Fit the feature engineering transforms
        self.preprocessing.fit(dataset=dataset)

        # Transform the input data to a valid forecast input
        input_data = self._prepare_input_data(dataset=dataset)

        # Fit the model
        if isinstance(self.forecaster, BaseForecaster):
            self.forecaster.fit(input_data=input_data)
        else:
            horizon_input_data = next(iter(input_data.values()), None)
            if horizon_input_data is None:
                raise ValueError("No data available for horizon forecasting.")

            self.forecaster.fit_horizon(input_data=horizon_input_data)

    def predict(
        self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        if not self.is_fitted:
            raise ModelNotFittedError(type(self.forecaster).__name__)

        # Transform the input data to a valid forecast input
        input_data = self._prepare_input_data(dataset=dataset, forecast_start=forecast_start)

        # Generate predictions
        if isinstance(self.forecaster, BaseForecaster):
            raw_forecasts = self.forecaster.predict(input_data=input_data)
        else:
            horizon_input_data = next(iter(input_data.values()), None)
            if horizon_input_data is None:
                raise ValueError("No data available for horizon forecasting.")

            raw_forecasts = self.forecaster.predict_horizon(input_data=horizon_input_data)

        return self.postprocessing.transform(raw_forecasts)
