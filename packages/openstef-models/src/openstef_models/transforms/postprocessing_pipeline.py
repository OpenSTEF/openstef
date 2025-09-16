# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import override

from openstef_core.datasets.transforms import ForecastTransform
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError


class PostprocessingPipeline(ForecastTransform):
    transforms: list[ForecastTransform]

    def __init__(self, transforms: list[ForecastTransform] | None = None) -> None:
        self.transforms = transforms or []

    @property
    def is_fitted(self) -> bool:
        return all(transform.is_fitted for transform in self.transforms)

    @override
    def fit(self, data: ForecastDataset) -> None:
        for transform in self.transforms:
            data = transform.fit_transform(data)

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        for transform in self.transforms:
            data = transform.transform(data)
        return data
