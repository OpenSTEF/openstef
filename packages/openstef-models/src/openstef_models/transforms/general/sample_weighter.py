# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Sample weighting for time series forecasting models.

Assigns importance weights to training samples based on target values,
emphasizing high-value periods for improved model performance on peak loads.
"""

import logging
from typing import override

import numpy as np
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms.dataset_transforms import TimeSeriesTransform
from openstef_models.transforms.general.scaler import StandardScaler


class SampleWeighter(BaseConfig, TimeSeriesTransform):
    """Transform that adds sample weights based on target variable magnitude.

    Computes weights using exponential scaling to emphasize high-value samples
    in model training. This is particularly useful for energy forecasting where
    accurate predictions during peak demand periods are more critical.

    The weighting scheme scales values relative to a high percentile (default 95th),
    applies an exponential transformation, and clips to a minimum floor value to
    ensure all samples contribute to training.

    Invariants:
        - Target column must exist in the dataset for weighting to be applied
        - Sample weights are always in the range [weight_floor, 1.0]
        - Rows with NaN target values receive default weight of 1.0
        - Transform is stateless and does not require fit()

    Example:
        >>> from datetime import timedelta
        >>> import pandas as pd
        >>> from openstef_core.testing import create_timeseries_dataset
        >>> from openstef_models.transforms.general import SampleWeighter
        >>> dataset = create_timeseries_dataset(
        ...     index=pd.date_range("2025-01-01", periods=5, freq="1h"),
        ...     load=[10.0, 50.0, 100.0, 200.0, 150.0],
        ...     sample_interval=timedelta(hours=1),
        ... )
        >>> transform = SampleWeighter()
        >>> result = transform.fit_transform(dataset)
        >>> result.data[["load", "sample_weight"]]
                              load  sample_weight
        timestamp
        2025-01-01 00:00:00   10.0       0.950413
        2025-01-01 01:00:00   50.0       0.537190
        2025-01-01 02:00:00  100.0       0.100000
        2025-01-01 03:00:00  200.0       1.000000
        2025-01-01 04:00:00  150.0       0.495868
    """

    weight_scale_percentile: int = Field(
        default=95,
        description="Percentile of target values used as scaling reference. "
        "Values are normalized relative to this percentile before weighting.",
    )
    weight_exponent: float = Field(
        default=1.0,
        description="Exponent applied to to scale the sample weights. "
        "0=uniform weights, 1=linear scaling, >1=stronger emphasis on high values.",
    )
    weight_floor: float = Field(
        default=0.1,
        description="Minimum weight value to ensure all samples contribute to training.",
    )
    target_column: str = Field(
        default="load",
        description="Column containing target values used for weight calculation.",
    )
    sample_weight_column: str = Field(
        default="sample_weight",
        description="Name of the column where computed weights will be stored.",
    )

    _scaler: StandardScaler = PrivateAttr(default_factory=StandardScaler)
    _is_fitted: bool = PrivateAttr(default=False)
    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        if self.target_column not in data.feature_names:
            self._logger.warning(
                "Target column '%s' not found in data features. Skipping sample weighting fit.", self.target_column
            )
            return

        target = np.asarray(data.data[self.target_column].dropna().values)
        self._scaler.fit(target.reshape(-1, 1))

        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        if self.target_column not in data.feature_names:
            self._logger.warning(
                "Target column '%s' not found in data features. Skipping sample weighting.", self.target_column
            )
            return data

        df = data.data.copy(deep=False)
        df[self.sample_weight_column] = 1.0  # default uniform weight

        # Set weights only for rows where target is not NaN
        mask = df[self.target_column].notna()
        target_series = df.loc[mask, self.target_column]

        # Normalize target values using the fitted scaler
        target = np.asarray(target_series.values, dtype=np.float64)
        target_scaled = self._scaler.transform(target.reshape(-1, 1)).flatten()

        df.loc[mask, self.sample_weight_column] = exponential_sample_weight(
            x=target_scaled,
            scale_percentile=self.weight_scale_percentile,
            exponent=self.weight_exponent,
            floor=self.weight_floor,
        )

        return data.copy_with(df)

    @override
    def features_added(self) -> list[str]:
        return [self.sample_weight_column]


def exponential_sample_weight(
    x: np.ndarray,
    scale_percentile: int = 95,
    exponent: float = 1.0,
    floor: float = 0.1,
) -> np.ndarray:
    """Calculate exponentially-scaled sample weights from target values.

    Normalizes values relative to a high percentile, applies exponential
    transformation, and clips to a minimum floor. This creates weights that
    emphasize high-value samples while ensuring all samples contribute.

    Args:
        x: Array of target values to compute weights from.
        scale_percentile: Percentile used as normalization reference (0-100).
            Values are divided by this percentile before scaling.
        exponent: Power to raise normalized values to. Controls emphasis strength:
            - 0: Uniform weights (all 1.0)
            - 1: Linear scaling proportional to values
            - >1: Stronger emphasis on high values
        floor: Minimum weight value. Ensures low-value samples still contribute.

    Returns:
        Array of weights in range [floor, 1.0] with same shape as input.

    Note:
        The function uses absolute values throughout, so negative inputs
        are weighted by their magnitude.
    """
    scaling_value = np.percentile(np.abs(x), scale_percentile)
    if scaling_value == 0:
        return np.full_like(x, fill_value=1.0)

    x_scaled = np.abs(x / scaling_value)
    x_weighted = np.abs(x_scaled) ** exponent
    return np.clip(x_weighted, a_min=floor, a_max=1.0)
