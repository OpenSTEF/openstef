# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for handling out-of-range feature values.

This module provides functionality to learn feature value bounds during training
and handle out-of-range values during inference, either by clipping them to the
learned bounds or replacing them with NaN.
"""

from typing import Literal, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection

type ClipMode = Literal["minmax", "standard"]
type OutlierAction = Literal["clip", "nan"]


class OutlierHandler(BaseConfig, TimeSeriesTransform):
    """Transform that handles out-of-range values for selected features.

    During fitting, this transform learns feature bounds from the training data.
    During transformation, values outside those bounds are either:

    - clipped to the learned bounds, or
    - replaced with NaN

    The bound definition depends on the selected mode:

    - ``minmax``: uses the observed minimum and maximum values
    - ``standard``: uses ``mean ± n_std * std``

    Examples:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general.outlier_handler import OutlierHandler
        >>> from openstef_models.utils.feature_selection import FeatureSelection
        >>>
        >>> training_data = pd.DataFrame({
        ...     "load": [100, 120, 110, 130, 125],
        ...     "temperature": [20, 22, 21, 23, 24],
        ... }, index=pd.date_range("2025-01-01", periods=5, freq="1h"))
        >>> training_dataset = TimeSeriesDataset(training_data, timedelta(hours=1))
        >>>
        >>> test_data = pd.DataFrame({
        ...     "load": [90, 140, 115],
        ...     "temperature": [19, 25, 22],
        ... }, index=pd.date_range("2025-01-06", periods=3, freq="1h"))
        >>> test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))
        >>>
        >>> # Default behavior: clip outliers
        >>> handler = OutlierHandler(
        ...     selection=FeatureSelection(include=["load", "temperature"]),
        ...     mode="minmax",
        ... )
        >>> handler.fit(training_dataset)
        >>> transformed_dataset = handler.transform(test_dataset)
        >>> transformed_dataset.data["load"].tolist()
        [100, 130, 115]
        >>> transformed_dataset.data["temperature"].tolist()
        [20, 24, 22]
        >>>
        >>> # Optional behavior: replace outliers with NaN
        >>> handler_nan = OutlierHandler(
        ...     selection=FeatureSelection(include=["load", "temperature"]),
        ...     mode="minmax",
        ...     outlier_action="nan",
        ... )
        >>> handler_nan.fit(training_dataset)
        >>> transformed_nan_dataset = handler_nan.transform(test_dataset)
        >>> transformed_nan_dataset.data["load"].tolist()
        [nan, nan, 115.0]
        >>> transformed_nan_dataset.data["temperature"].tolist()
        [nan, nan, 22.0]
    """

    selection: FeatureSelection = Field(default=FeatureSelection.ALL, description="Features to handle.")
    mode: ClipMode = Field(
        default="minmax",
        description="Boundary mode: 'minmax' uses observed min/max, 'standard' uses mean ± n_std * std.",
    )
    outlier_action: OutlierAction = Field(
        default="clip",
        description=(
            "How to handle out-of-range values: 'clip' clips the values to bounds, "
            "'nan' replaces them with NaN."
        ),
    )
    n_std: float = Field(
        default=2.0,
        gt=0.0,
        description="Number of standard deviations to use when mode='standard'.",
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
        selected_data = data.data.reindex(features, axis=1)

        if self.mode == "minmax":
            self._feature_mins = selected_data.min()
            self._feature_maxs = selected_data.max()
        else:  # mode == "standard"
            self._feature_means = selected_data.mean()
            self._feature_stds = selected_data.std()

        self._is_fitted = True

    def _get_bounds(self, features: list[str]) -> tuple[pd.Series, pd.Series]:
        if self.mode == "minmax":
            lower_bound = self._feature_mins.reindex(features)
            upper_bound = self._feature_maxs.reindex(features)
        else:  # mode == "standard"
            mean_aligned = self._feature_means.reindex(features)
            std_aligned = self._feature_stds.reindex(features)
            lower_bound = mean_aligned - self.n_std * std_aligned
            upper_bound = mean_aligned + self.n_std * std_aligned

        return lower_bound, upper_bound

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        features = self.selection.resolve(data.feature_names)
        transformed_data = data.data.copy(deep=False)
        lower_bound, upper_bound = self._get_bounds(features)

        if self.outlier_action == "clip":
            transformed_data[features] = data.data[features].clip(
                lower=lower_bound,
                upper=upper_bound,
                axis=1,
            )
        else:  # outlier_action == "nan"
            feature_data = data.data[features].copy()

            below_lower = feature_data < np.asarray(lower_bound)
            above_upper = feature_data > np.asarray(upper_bound)
            out_of_range_mask = below_lower | above_upper

            transformed_data[features] = feature_data.where(~out_of_range_mask)

        return TimeSeriesDataset(data=transformed_data, sample_interval=data.sample_interval)

    @override
    def features_added(self) -> list[str]:
        return []
