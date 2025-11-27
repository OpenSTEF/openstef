# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for clipping feature values to observed ranges.

This module provides functionality to clip feature values to their observed
minimum and maximum ranges during training, preventing out-of-range values
during inference and improving model robustness.
"""

from typing import override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class Flagger(BaseConfig, TimeSeriesTransform):
    """Transform that flags specified features to their observed min and max values.

    This transform flags the peaks for the metalearner to know when to expect outliers and
    extrapolate from its training set.


    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_meta.transforms import Flagger
        >>> from openstef_models.utils.feature_selection import FeatureSelection
        >>> # Create sample training dataset
        >>> training_data = pd.DataFrame({
        ...     'load': [100, 90, 110],
        ...     'temperature': [19, 20, 21]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='1h'))
        >>> training_dataset = TimeSeriesDataset(training_data, timedelta(hours=1))
        >>> test_data = pd.DataFrame({
        ...     'load': [90, 140, 100],
        ...     'temperature': [18, 20, 22]
        ... }, index=pd.date_range('2025-01-06', periods=3,
        ... freq='1h'))
        >>> test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))
        >>> # Initialize and apply transform
        >>> flagger = Flagger(selection=FeatureSelection(include=['load', 'temperature']))
        >>> flagger.fit(training_dataset)
        >>> transformed_dataset = flagger.transform(test_dataset)
        >>> transformed_dataset.data['load'].tolist()
        [0, 0, 1]
        >>> transformed_dataset.data['temperature'].tolist()
        [0, 1, 0]

    """

    selection: FeatureSelection = Field(default=FeatureSelection.ALL, description="Features to flag.")

    _feature_mins: pd.Series = PrivateAttr(default_factory=pd.Series)
    _feature_maxs: pd.Series = PrivateAttr(default_factory=pd.Series)
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        features = self.selection.resolve(data.feature_names)
        self._feature_mins = data.data.reindex(features, axis=1).min()
        self._feature_maxs = data.data.reindex(features, axis=1).max()
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        features = self.selection.resolve(data.feature_names)
        transformed_data = data.data.copy(deep=False).loc[:, features]

        # compute min & max of the features
        min_aligned = self._feature_mins.reindex(features)
        max_aligned = self._feature_maxs.reindex(features)

        outside = (transformed_data[features] <= min_aligned) | (transformed_data[features] >= max_aligned)
        transformed_data = (~outside).astype(int)

        return TimeSeriesDataset(data=transformed_data, sample_interval=data.sample_interval)

    @override
    def features_added(self) -> list[str]:
        return []
