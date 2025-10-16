# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for scaling features in time series data.

This module provides data scaling functionality using various scaling methods
from scikit-learn to normalize and standardize features in time series datasets
for improved machine learning model performance.
"""

from typing import Any, Literal, Self, cast, override

from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingExtraError, TransformNotFittedError
from openstef_core.mixins import State
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection

try:
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
except ImportError as e:
    raise MissingExtraError("sklearn", package="openstef-models") from e


type ScalingMethod = Literal["min-max", "max-abs", "standard", "robust"]


class Scaler(BaseConfig, TimeSeriesTransform):
    """Transform that scales time series data using various scikit-learn scaling methods.

    Available methods include:
        - MinMaxScaler: Scales features based on min/max of training set (between 0 and 1).
        - MaxAbs: Scales features by their maximum absolute value (between -1 and 1).
        - Standard: Standardizes features by removing the mean and scaling to unit variance (0 mean, 1 std).
        - Robust: Scales features using statistics that are robust to outliers: median and IQR.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import Scaler
        >>>
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'load': [100, 200, 300],
        ...     'temperature': [20, 25, 30]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Initialize and apply transform
        >>> scaler = Scaler(method="standard")
        >>> scaler.fit(dataset)
        >>> transformed_dataset = scaler.transform(dataset)
        >>> abs(float(transformed_dataset.data['load'].mean().round(6)))
        0.0
        >>> # use ddof=0 to get population std (as used by StandardScaler)
        >>> float(transformed_dataset.data['load'].std(ddof=0).round(6))
        1.0
        >>> abs(float(transformed_dataset.data['temperature'].mean().round(6)))
        0.0
        >>> float(transformed_dataset.data['temperature'].std(ddof=0).round(6))
        1.0
    """

    method: ScalingMethod = Field(default="standard", description="Scaling method to use.")
    selection: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description="Features to scale.",
    )

    _scaler: MinMaxScaler | MaxAbsScaler | StandardScaler | RobustScaler = PrivateAttr()
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def model_post_init(self, context: Any) -> None:
        match self.method:
            case "min-max":
                self._scaler = MinMaxScaler()
            case "max-abs":
                self._scaler = MaxAbsScaler()
            case "standard":
                self._scaler = StandardScaler()
            case "robust":
                self._scaler = RobustScaler()

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the scaler to the input time series data.

        Args:
            data: Time series dataset.
        """
        features = self.selection.resolve(data.feature_names)
        self._scaler.fit(data.data[features])  # type: ignore[reportUnknownMemberType]
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise TransformNotFittedError(self.__class__.__name__)

        features = self.selection.resolve(data.feature_names)
        scaled_data = data.data.copy()
        scaled_data[features] = self._scaler.transform(scaled_data[features])

        return TimeSeriesDataset(
            data=scaled_data,
            sample_interval=data.sample_interval,
        )

    @override
    def to_state(self) -> State:
        return cast(
            State,
            {
                "config": self.model_dump(mode="json"),
                "scaler": self._scaler.__getstate__(),  # pyright: ignore[reportUnknownMemberType]
                "is_fitted": self._is_fitted,
            },
        )

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, Any], state)
        instance = self.model_validate(state["config"])
        instance._scaler.__setstate__(state["scaler"])  # pyright: ignore[reportUnknownMemberType]  # noqa: SLF001
        instance._is_fitted = state["is_fitted"]  # noqa: SLF001
        return instance

    @override
    def features_added(self) -> list[str]:
        return []
