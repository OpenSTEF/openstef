# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for scaling features in time series data.

This module provides data scaling functionality using various scaling methods
from scikit-learn to normalize and standardize features in time series datasets
for improved machine learning model performance.
"""

from typing import Any, Literal, cast, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform
from openstef_core.exceptions import MissingExtraError, TransformNotFittedError
from openstef_core.types import LeadTime
from openstef_models.transforms.horizon_split_transform import concat_horizon_datasets_rowwise

try:
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
except ImportError as e:
    raise MissingExtraError("transforms") from e


type ScalingMethod = Literal["min-max", "max-abs", "standard", "robust"]


class ScalerTransform(BaseConfig, TimeSeriesTransform):
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
        >>> from openstef_models.transforms.general import ScalerTransform
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'temperature': [20, 22, 21, 23, 24]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Initialize and apply transform
        >>> scaler = ScalerTransform(method="standard")
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

    method: ScalingMethod = Field(description="Scaling method to use.")

    _scaler: MinMaxScaler | MaxAbsScaler | StandardScaler | RobustScaler = PrivateAttr()
    _is_fitted: bool = PrivateAttr(default=False)

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
    def fit_horizons(self, data: dict[LeadTime, TimeSeriesDataset]) -> None:
        flat_data = concat_horizon_datasets_rowwise(data)
        return self.fit(flat_data)

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the scaler to the input time series data.

        Args:
            data: Time series dataset.
        """
        self._scaler.fit(data.data)  # type: ignore[reportUnknownMemberType]
        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise TransformNotFittedError(self.__class__.__name__)

        scaled_data = pd.DataFrame(
            data=cast(pd.DataFrame, self._scaler.transform(data.data)),
            columns=data.data.columns,
            index=data.data.index,
        )
        return TimeSeriesDataset(
            data=scaled_data,
            sample_interval=data.sample_interval,
        )
