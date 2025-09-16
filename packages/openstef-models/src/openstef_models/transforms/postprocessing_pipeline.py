# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import override

from openstef_core.datasets.transforms import ForecastTransform
from openstef_core.datasets.validated_datasets import ForecastDataset


class PostprocessingPipeline(ForecastTransform):
    transforms: list[ForecastTransform]

    def __init__(self, transforms: list[ForecastTransform] | None = None) -> None:
        self.transforms = transforms or []

    @override
    def fit(self, data: ForecastDataset) -> None:
        for transform in self.transforms:
            data = transform.fit_transform(data)

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        for transform in self.transforms:
            data = transform.transform(data)
        return data
