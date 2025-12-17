# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for dropping for dropping features from dataset based on FeatureSelection.

This transform allows selecting a subset of features from a TimeSeriesDataset based on a specified
FeatureSelection strategy. It can be used to exclude certain features before model training
or inference.
"""

from typing import override

from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class Selector(BaseConfig, TimeSeriesTransform):
    """Selects features based on FeatureSelection.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import Selector
        >>> from openstef_models.utils.feature_selection import FeatureSelection
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame(
        ...     {
        ...         "load": [100.0, 110.0, 120.0],
        ...         "temperature": [20.0, 22.0, 23.0],
        ...         "humidity": [60.0, 65.0, 70.0],
        ...     },
        ...     index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        ... )
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Select specific features
        >>> selector = Selector(selection=FeatureSelection(include={'load', 'temperature'}))
        >>> transformed = selector.transform(dataset)
        >>> transformed.feature_names
        ['load', 'temperature']
    """

    selection: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description="Feature selection for efficient model specific preprocessing.",
    )
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        if (
            isinstance(data, ForecastInputDataset)
            and self.selection.include is not None
            and (data.target_column not in self.selection.include)
        ):
            self.selection.include.add(data.target_column)

        self._is_fitted = True

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
