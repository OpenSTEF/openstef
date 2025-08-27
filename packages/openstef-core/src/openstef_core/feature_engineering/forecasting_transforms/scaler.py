# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for scaling features in time series data.

This module provides data scaling functionality using various scaling methods
from scikit-learn to normalize and standardize features in time series datasets
for improved machine learning model performance.
"""

from enum import StrEnum

import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

try:
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for the Scaler transform. Please install it via "
        "`uv sync --group ml --package openstef-core` or `uv sync --all-groups --package openstef-core`."
    ) from e


class ScalingMethod(StrEnum):
    """Scaling methods from sklearn.preprocessing."""

    MinMax = "min-max"
    MaxAbs = "max-abs"
    Standard = "standard"
    Robust = "robust"


class Scaler(TimeSeriesTransform):
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
        >>> from openstef_core.feature_engineering.forecasting_transforms.scaler import Scaler, ScalingMethod
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'temperature': [20, 22, 21, 23, 24]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Initialize and apply transform
        >>> scaler = Scaler(method=ScalingMethod.Standard)
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

    def __init__(self, method: ScalingMethod):
        """Initialize the Scaler transform with the scaler method.

        Args:
            method: Scaling method to use.

        Raises:
            ValueError: If an unsupported scaling method is provided.
        """
        match method:
            case ScalingMethod.MinMax:
                self.scaler = MinMaxScaler()
            case ScalingMethod.MaxAbs:
                self.scaler = MaxAbsScaler()
            case ScalingMethod.Standard:
                self.scaler = StandardScaler()
            case ScalingMethod.Robust:
                self.scaler = RobustScaler()
            case _:
                msg = f"Unsupported normalization method: {method}"
                raise ValueError(msg)

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the scaler to the input time series data.

        Args:
            data: Time series dataset.
        """
        self.scaler.fit(data.data)  # type: ignore[reportUnknownMemberType]

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data using the fitted scaler.

        Args:
            data: Time series dataset to transform.

        Returns:
            A new TimeSeriesDataset instance containing the scaled data.
        """
        scaled_data = pd.DataFrame(
            data=self.scaler.transform(data.data),  # type: ignore[reportUnknownMemberType]
            columns=data.data.columns,
            index=data.data.index,
        )
        return TimeSeriesDataset(
            data=scaled_data,
            sample_interval=data.sample_interval,
        )
