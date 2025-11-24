# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for clipping feature values to observed ranges.

This module provides functionality to clip feature values to their observed
minimum and maximum ranges during training, preventing out-of-range values
during inference and improving model robustness.
"""

from typing import Literal, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection

type ClipMode = Literal["minmax", "standard"]


class DistributionTransform(BaseConfig, TimeSeriesTransform):
    """Transform dataframe to (robust) percentage of min-max of training data.

    Useful to determine whether datadrift has occured.
    Can be used as a feature for learning sample weights in meta models.
    """

    robust_threshold: float = Field(
        default=2.0,
        description="Percentage of observations to ignore when determing percentage. (Single sided)",
    )

    _feature_mins: pd.Series = PrivateAttr(default_factory=pd.Series)
    _feature_maxs: pd.Series = PrivateAttr(default_factory=pd.Series)
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        self._feature_mins = data.data.min(axis=0)
        self._feature_maxs = data.data.max(axis=0)
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Apply min-max scaling to each feature based on fitted min and max
        transformed_data = (data.data - self._feature_mins) / (self._feature_maxs - self._feature_mins)

        return TimeSeriesDataset(data=transformed_data, sample_interval=data.sample_interval)

    @override
    def features_added(self) -> list[str]:
        return []
