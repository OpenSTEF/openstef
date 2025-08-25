# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum

import pandas as pd

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

try:
    import sklearn
except ImportError:
    raise ImportError(
        "scikit-learn is required for the Scaler transform. Please install it via `uv sync --group ml --package openstef-core` or `uv sync --all-groups --package openstef-core`."
    )


class ScalerMode(StrEnum):
    """Scaling methods from sklearn.preprocessing."""

    MinMax = "min-max"
    MaxAbs = "max-abs"
    Standard = "standard"
    Robust = "robust"


class ScalerConfig(BaseConfig):
    """Configuration for the Scaler transform."""

    method: ScalerMode


class Scaler(TimeSeriesTransform):
    """Transform that scales time series data using various scikit-learn
    scaling methods. Available methods include:
        - MinMaxScaler: Scales features based on min/max of training set (between 0 and 1).
        - MaxAbs: Scales features by their maximum absolute value (between -1 and 1).
        - Standard: Standardizes features by removing the mean and scaling to unit variance (0 mean, 1 std).
        - Robust: Scales features using statistics that are robust to outliers: median and IQR.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.forecasting_transforms.scaler import Scaler, ScalerConfig, ScalerMode
        >>>
        >>> # Create sample dataset
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'temperature': [20, 22, 21, 23, 24]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Initialize and apply transform
        >>> config = ScalerConfig(method=ScalerMode.Standard)
        >>> scaler = Scaler(config)
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

    def __init__(self, config: ScalerConfig):
        """Initialize the Scaler transform with the scaler
        specified in the configuration.
        """
        self.config = config
        match self.config.method:
            case ScalerMode.MinMax:
                from sklearn.preprocessing import MinMaxScaler

                self.scaler = MinMaxScaler()
            case ScalerMode.MaxAbs:
                from sklearn.preprocessing import MaxAbsScaler

                self.scaler = MaxAbsScaler()
            case ScalerMode.Standard:
                from sklearn.preprocessing import StandardScaler

                self.scaler = StandardScaler()
            case ScalerMode.Robust:
                from sklearn.preprocessing import RobustScaler

                self.scaler = RobustScaler()
            case _:
                raise ValueError(f"Unsupported normalization method: {self.config.method}")

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the scaler to the input time series data.

        Args:
            data: Time series dataset.
        """
        self.scaler.fit(data.data)

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data using the fitted scaler.

        Args:
            data: Time series dataset to transform.

        Returns:
            A new TimeSeriesDataset instance containing the scaled data.
        """
        scaled_data = pd.DataFrame(
            data=self.scaler.transform(data.data),
            columns=data.data.columns,
            index=data.data.index,
        )
        return TimeSeriesDataset(
            data=scaled_data,
            sample_interval=data.sample_interval,
        )
