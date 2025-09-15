from datetime import datetime
from openstef_core.datasets import ForecastInputDataset, TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_models.feature_engineering import FeaturePipeline
from openstef_models.models.forecasting.mixins import BaseForecaster, BaseHorizonForecaster


class ForecasterPipeline:
    feature_engineering: FeaturePipeline
    model: BaseForecaster | BaseHorizonForecaster
    target_column: str
    callbacks: list[None]

    def fit(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        input_data = self.feature_engineering.transform(dataset=dataset)

        forecast_input_data = {
            key: ForecastInputDataset.from_timeseries_dataset(
                dataset=value,
                target_column=self.target_column,
                forecast_start=None,
            )
            for key, value in input_data.items()
        }

        if isinstance(self.model, BaseForecaster):
            self.model.fit(input_data=forecast_input_data)
        else:
            forecast_horizon_input_data = next(iter(forecast_input_data.values()), None)
            if forecast_horizon_input_data is None:
                raise ValueError("No data available for horizon forecasting.")

            self.model.fit_horizon(input_data=forecast_horizon_input_data)

    def predict(
        self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        input_data = self.feature_engineering.transform(dataset=dataset)

        forecast_input_data = {
            key: ForecastInputDataset.from_timeseries_dataset(
                dataset=value,
                target_column=self.target_column,
                forecast_start=forecast_start,
            )
            for key, value in input_data.items()
        }

        if isinstance(self.model, BaseForecaster):
            forecasts = self.model.predict(input_data=forecast_input_data)
        else:
            forecast_horizon_input_data = next(iter(forecast_input_data.values()), None)
            if forecast_horizon_input_data is None:
                raise ValueError("No data available for horizon forecasting.")

            forecasts = self.model.predict_horizon(input_data=forecast_horizon_input_data)

        return forecasts
