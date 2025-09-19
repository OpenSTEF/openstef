# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import override

from openstef_core.base_model import BaseModel
from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_models.models import ComponentSplittingModel
from openstef_models.models.mixins import ComponentSplitterConfig, ComponentSplitterMixin


class ComponentSplitWorkflow(BaseModel, ComponentSplitterMixin):
    model: ComponentSplittingModel

    @property
    @override
    def config(self) -> ComponentSplitterConfig:
        return self.model.config

    @property
    @override
    def is_fitted(self) -> bool:
        return self.model.is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        self.model.fit(data)

    @override
    def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
        return self.model.predict(data)
