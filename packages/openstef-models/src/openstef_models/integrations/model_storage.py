# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

from openstef_models.models.forecasting_model import ForecastingModel

type ModelIdentifier = str


class ModelStorage(ABC):
    @abstractmethod
    def load_model(self, model_id: ModelIdentifier) -> ForecastingModel: ...

    @abstractmethod
    def save_model(self, model_id: ModelIdentifier, model: ForecastingModel) -> None: ...
