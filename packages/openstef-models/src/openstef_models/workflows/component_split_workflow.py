# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0


from typing import Self

from pydantic import Field

from openstef_core.base_model import BaseModel
from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.mixins import ModelIdentifier, ModelStorage
from openstef_models.mixins.callbacks import PredictorCallback
from openstef_models.models import ComponentSplittingModel


class ComponentSplitCallback(PredictorCallback["ComponentSplitWorkflow", TimeSeriesDataset, EnergyComponentDataset]):
    pass


class ComponentSplitWorkflow(BaseModel):
    model: ComponentSplittingModel = Field(...)
    callbacks: ComponentSplitCallback = Field(default_factory=ComponentSplitCallback)
    storage: ModelStorage | None = Field(...)
    model_id: ModelIdentifier = Field(...)

    def fit(self, data: TimeSeriesDataset) -> None:
        self.callbacks.on_fit_start(workflow=self, data=data)

        self.model.fit(data=data)

        self.callbacks.on_fit_end(workflow=self, data=data)

        if self.storage is not None:
            self.storage.save_model_state(model_id=self.model_id, model=self.model)

    def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
        if not self.model.is_fitted:
            raise NotFittedError(type(self.model).__name__)

        self.callbacks.on_predict_start(workflow=self, data=data)

        prediction = self.model.predict(data=data)

        self.callbacks.on_predict_end(workflow=self, data=data, forecasts=prediction)

        return prediction

    @classmethod
    def from_storage(
        cls,
        model_id: ModelIdentifier,
        model: ComponentSplittingModel,
        storage: ModelStorage,
        callbacks: ComponentSplitCallback | None = None,
    ) -> Self:
        """Create a workflow by loading a model from storage with optional fallback.

        Attempts to load a model from storage. If the model is not found and a
        default factory is provided, creates a new model using the factory.

        Args:
            model_id: Identifier for the model to load from storage.
            model: The model to use for training and prediction.
            storage: Model storage system to load from.
            callbacks: Optional callback handler for the workflow.

        Returns:
            New workflow instance with the loaded or created model.
        """
        model = storage.load_model_state(model_id=model_id, model=model)
        return cls(model=model, callbacks=callbacks or ComponentSplitCallback(), storage=storage, model_id=model_id)
