# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for dropping rows containing NaN values.

This module provides functionality to drop rows containing NaN values in selected
columns, useful for data cleaning and ensuring complete cases for model training.
"""

from typing import override

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class Selector(BaseConfig, TimeSeriesTransform):
    """Selects features based on FeatureSelection."""

    selection: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description="Feature selection for efficient model specific preprocessing.x",
    )

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        if (
            isinstance(data, ForecastInputDataset)
            and self.selection.include is not None
            and (data.target_column not in self.selection.include)
        ):
            self.selection.include.add(data.target_column)

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        features = self.selection.resolve(data.feature_names)

        transformed_data = data.data.drop(
            columns=[col for col in data.feature_names if col not in features]
        )

        return data.copy_with(data=transformed_data, is_sorted=True)

    @override
    def features_added(self) -> list[str]:
        return []
