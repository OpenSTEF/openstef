# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0


from openstef_core.base_model import BaseModel
from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_models.models import ComponentSplittingModel


class ComponentSplitWorkflow(BaseModel):
    model: ComponentSplittingModel

    def fit(self, data: TimeSeriesDataset) -> None:
        self.model.fit(data)

    def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
        return self.model.predict(data)
