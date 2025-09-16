# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime
from typing import Self

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import ModelNotFoundError, NotFittedError
from openstef_models.integrations.callbacks import ForecastingCallback
from openstef_models.integrations.model_storage import ModelIdentifier, ModelStorage
from openstef_models.models.forecasting_model import ForecastingModel


class ForecastingWorkflow:
    model: ForecastingModel
    callbacks: ForecastingCallback
    storage: ModelStorage | None
    model_id: ModelIdentifier

    def __init__(
        self,
        model: ForecastingModel,
        callbacks: ForecastingCallback | None = None,
        storage: ModelStorage | None = None,
    ) -> None:
        self.model = model
        self.callbacks = callbacks or ForecastingCallback()
        self.storage = storage

    def fit(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        self.callbacks.on_before_fit(pipeline=self, dataset=dataset)

        self.model.fit(dataset=dataset)

        self.callbacks.on_after_fit(pipeline=self, dataset=dataset)

    def predict(
        self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        if not self.model.is_fitted:
            raise NotFittedError(type(self.model).__name__)

        self.callbacks.on_before_predict(pipeline=self, dataset=dataset)

        forecasts = self.model.predict(dataset=dataset, forecast_start=forecast_start)

        self.callbacks.on_after_predict(pipeline=self, dataset=dataset, forecasts=forecasts)

        return forecasts

    @classmethod
    def from_storage(
        cls,
        model_id: ModelIdentifier,
        storage: ModelStorage,
        callbacks: ForecastingCallback | None = None,
        default_model: ForecastingModel | None = None,
    ) -> Self:
        try:
            model = storage.load_model(model_id=model_id)
        except ModelNotFoundError:
            if default_model is None:
                raise

            model = default_model

        return cls(model=model, callbacks=callbacks, storage=storage)


__all__ = ["ForecastingWorkflow"]
