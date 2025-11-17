# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Kalman Filter Transforms for Time Series Data pre and post-processing.

This class provides implementations of Kalman Smoothing as both a preprocessor
and postprocessor for time series datasets. The Kalman Smoother helps reduce noise
"""

from collections.abc import Iterable
from typing import override

import pandas as pd
from pydantic import Field
from sktime.transformations.series.kalman_filter import (
    KalmanFilterTransformerFP,
)

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, TimeSeriesDataset
from openstef_core.mixins import Transform
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class BaseKalman(BaseConfig):
    """Base class for Kalman Smoothing transforms."""

    selection: FeatureSelection = Field(default=FeatureSelection.ALL, description="Columns to smooth")
    state_dim: int = Field(
        default=1,
        description="Kalman filter state dimension (1 = per-column independent)",
    )

    @staticmethod
    def _run_kalman_smoother(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        features_list = list(features)
        if not features_list:
            return df

        kf = KalmanFilterTransformerFP(state_dim=len(features_list))
        out = df.copy(deep=True)
        out[features_list] = kf.fit_transform(X=df[features_list])  # type: ignore[assignment]
        return out


# Preprocessor: operates on TimeSeriesDataset
class KalmanPreprocessor(BaseKalman, TimeSeriesTransform):
    """Apply Kalman Smoothing to time series data to reduce noise and improve temporal consistency.

    Example:
        >>> from datetime import timedelta
        >>> import pandas as pd
        >>> from openstef_core.testing import create_timeseries_dataset
        >>> from openstef_models.transforms.general import KalmanPreprocessor
        >>> dataset = create_timeseries_dataset(
        ...     index=pd.date_range("2025-01-01", periods=5, freq="1h"),
        ...     load=[10.0, 50.0, 100.0, 200.0, 150.0],
        ...     sample_interval=timedelta(hours=1),
        ... )
        >>> transform = KalmanPreprocessor()
        >>> result = transform.fit_transform(dataset)
        >>> result.data
                                   load
        timestamp
        2025-01-01 00:00:00    6.666667
        2025-01-01 01:00:00   33.750000
        2025-01-01 02:00:00   74.761905
        2025-01-01 03:00:00  152.181818
        2025-01-01 04:00:00  150.833333
    """

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        # stateless: nothing to do
        return None

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        features: list[str] = list(self.selection.resolve(data.feature_names))
        # restrict to numeric
        numeric = data.data.select_dtypes(include=["number"]).columns.tolist()
        features = [f for f in features if f in numeric]
        if not features:
            return data
        df = data.data.copy(deep=True)
        df_sm = self._run_kalman_smoother(df, features)
        return data.copy_with(data=df_sm, is_sorted=True)

    @override
    def features_added(self) -> list[str]:
        # Preprocessor doesn't add columns
        return []


# Postprocessor: operates on ForecastDataset quantiles
class KalmanPostprocessor(BaseKalman, Transform[ForecastDataset, ForecastDataset]):
    """Apply Kalman Smoothing to quantile forecasts to reduce noise and improve temporal consistency.

    Example:
        >>> from datetime import timedelta
        >>> import pandas as pd
        >>> import numpy as np
        >>> from openstef_core.datasets.validated_datasets import ForecastDataset
        >>> from openstef_models.transforms.general import KalmanPostprocessor
        >>> forecast_data = pd.DataFrame({
        ...     'load': [100, np.nan],
        ...     'quantile_P10': [90, 95],
        ...     'quantile_P50': [100, 110],
        ...     'quantile_P90': [115, 125]
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='h'))
        >>> dataset = ForecastDataset(forecast_data, timedelta(hours=1))
        >>> transform = KalmanPostprocessor()
        >>> result = transform.fit_transform(dataset)
        >>> result.data
                            load  quantile_P10  quantile_P50  quantile_P90
        timestamp
        2025-01-01 00:00:00  100.0        60.000     66.666667     76.666667
        2025-01-01 01:00:00    NaN        81.875     93.750000    106.875000
    """

    monotonic: bool = Field(
        default=True,
        description="Enforce non-crossing quantiles after smoothing",
    )

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: ForecastDataset) -> None:
        return None

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        quantile_columns = [q.format() for q in sorted(data.quantiles)]
        df = data.data.copy(deep=True)
        df_sm = self._run_kalman_smoother(df, quantile_columns)
        return ForecastDataset.from_timeseries(data.copy_with(data=df_sm, is_sorted=True))


__all__ = ["BaseKalman", "KalmanPostprocessor", "KalmanPreprocessor"]
