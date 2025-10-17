# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for clipping feature values to observed ranges.

This module provides functionality to clip feature values to their observed
minimum and maximum ranges during training, preventing out-of-range values
during inference and improving model robustness.
"""

from typing import Any, Literal, Self, cast, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection

type ClipMode = Literal["minmax", "standard"]


class Clipper(BaseConfig, TimeSeriesTransform):
    """Transform that clips specified features to their observed min and max values.

    This transform learns the minimum and maximum values of specified features
    during the fit phase and clips any values outside this range during the transform phase.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import Clipper
        >>>
        >>> # Create sample training dataset
        >>> training_data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'temperature': [20, 22, 21, 23, 24]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> training_dataset = TimeSeriesDataset(training_data, timedelta(hours=1))
        >>> test_data = pd.DataFrame({
        ...     'load': [90, 140, 115],
        ...     'temperature': [19, 25, 22]
        ... }, index=pd.date_range('2025-01-06', periods=3,
        ... freq='1h'))
        >>> test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))
        >>> # Initialize and apply transform
        >>> clipper = Clipper(selection=FeatureSelection(include=['load', 'temperature']), mode='minmax')
        >>> clipper.fit(training_dataset)
        >>> transformed_dataset = clipper.transform(test_dataset)
        >>> clipper._feature_mins.to_dict()
        {'load': 100, 'temperature': 20}
        >>> clipper._feature_maxs.to_dict()
        {'load': 130, 'temperature': 24}
        >>> transformed_dataset.data['load'].tolist()
        [100, 130, 115]
        >>> transformed_dataset.data['temperature'].tolist()
        [20, 24, 22]

    """

    selection: FeatureSelection = Field(default=FeatureSelection.ALL, description="Features to clip.")
    mode: ClipMode = Field(
        default="minmax",
        description="Clipping mode: 'minmax' clips to observed min/max, 'standard' clips to mean ± n_std * std",
    )
    n_std: float = Field(
        default=2.0,
        gt=0.0,
        description="Number of standard deviations to clip at (only used when mode='standard')",
    )

    _feature_mins: pd.Series = PrivateAttr(default_factory=pd.Series)
    _feature_maxs: pd.Series = PrivateAttr(default_factory=pd.Series)
    _feature_means: pd.Series = PrivateAttr(default_factory=pd.Series)
    _feature_stds: pd.Series = PrivateAttr(default_factory=pd.Series)
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        features = self.selection.resolve(data.feature_names)
        if self.mode == "minmax":
            self._feature_mins = data.data.reindex(features, axis=1).min()
            self._feature_maxs = data.data.reindex(features, axis=1).max()
        else:  # mode == "standard"
            self._feature_means = data.data.reindex(features, axis=1).mean()
            self._feature_stds = data.data.reindex(features, axis=1).std()
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        features = self.selection.resolve(data.feature_names)
        transformed_data = data.data.copy()

        if self.mode == "minmax":
            min_aligned = self._feature_mins.reindex(features)
            max_aligned = self._feature_maxs.reindex(features)
            transformed_data[features] = data.data[features].clip(lower=min_aligned, upper=max_aligned, axis=1)
        else:  # mode == "standard"
            mean_aligned = self._feature_means.reindex(features)
            std_aligned = self._feature_stds.reindex(features)
            lower_bound = mean_aligned - self.n_std * std_aligned
            upper_bound = mean_aligned + self.n_std * std_aligned
            transformed_data[features] = data.data[features].clip(lower=lower_bound, upper=upper_bound, axis=1)

        return TimeSeriesDataset(data=transformed_data, sample_interval=data.sample_interval)

    @override
    def to_state(self) -> State:
        state: dict[str, Any] = {
            "config": self.model_dump(mode="json"),
            "is_fitted": self._is_fitted,
            "mode": self.mode,
        }
        if self.mode == "minmax":
            state["feature_mins"] = self._feature_mins
            state["feature_maxs"] = self._feature_maxs
        else:  # mode == "standard"
            state["feature_means"] = self._feature_means
            state["feature_stds"] = self._feature_stds
        return cast(State, state)

    @override
    def from_state(self, state: State) -> Self:
        state_dict = cast(dict[str, Any], state)
        instance = self.model_validate(state_dict["config"])

        if instance.mode == "minmax":
            instance._feature_mins = state_dict["feature_mins"]  # noqa: SLF001
            instance._feature_maxs = state_dict["feature_maxs"]  # noqa: SLF001
        else:  # mode == "standard"
            instance._feature_means = state_dict["feature_means"]  # noqa: SLF001
            instance._feature_stds = state_dict["feature_stds"]  # noqa: SLF001

        instance._is_fitted = state_dict["is_fitted"]  # noqa: SLF001
        return instance

    @override
    def features_added(self) -> list[str]:
        return []
