# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Sample weighting for time series forecasting models.

Assigns importance weights to training samples based on target values,
emphasizing high-value periods for improved model performance on peak loads.
"""

import logging
from typing import Literal, override

import numpy as np
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.transforms.dataset_transforms import TimeSeriesTransform
from openstef_models.transforms.general.scaler import StandardScaler


class SampleWeighter(BaseConfig, TimeSeriesTransform):
    """Transform that adds sample weights based on target variable distribution.

    Supports two weighting methods:

    - exponential (default): Scales weights by target magnitude relative to a
      high percentile. Useful for emphasizing high-value samples like peak loads.
    - inverse_frequency: Weights samples inversely proportional to their frequency
      in the target distribution. Rare values receive higher weights.

    Both methods clip weights to [weight_floor, 1.0] to ensure all samples contribute.

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
        2025-01-01 00:00:00   10.0       0.100000
        2025-01-01 01:00:00   50.0       0.263158
        2025-01-01 02:00:00  100.0       0.526316
        2025-01-01 03:00:00  200.0       1.000000
        2025-01-01 04:00:00  150.0       0.789474
    """

    method: Literal["exponential", "inverse_frequency"] = Field(
        default="exponential",
        description="Weighting method: 'exponential' scales by magnitude, 'inverse_frequency' by rarity.",
    )
    weight_scale_percentile: int = Field(
        default=95,
        description="[exponential method only] Percentile of target values used as scaling reference.",
    )
    weight_exponent: float = Field(
        default=1.0,
        description="[exponential method only] Exponent for scaling: 0=uniform, 1=linear, >1=stronger emphasis.",
    )
    n_bins: int = Field(
        default=50,
        description="[inverse_frequency method only] Number of equal-width histogram bins for frequency estimation.",
    )
    dampening_exponent: float = Field(
        default=0.5,
        description="[inverse_frequency method only] Exponent in [0,1] applied to inverse frequency to compress range.",
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
    normalize_target: bool = Field(
        default=False,
        description="Whether to normalize target values using StandardScaler before weighting.",
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
        if self.target_column not in data.feature_names:
            self._logger.warning(
                "Target column '%s' not found in data features. Skipping sample weighting.", self.target_column
            )
            return data

        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        df = data.data.copy(deep=False)
        df[self.sample_weight_column] = 1.0  # default uniform weight

        # Set weights only for rows where target is not NaN
        mask = df[self.target_column].notna()
        target_series = df.loc[mask, self.target_column]
        target = np.asarray(target_series.values, dtype=np.float64)

        # Normalize target values using the fitted scaler
        if self.normalize_target and target.size > 0:
            target = self._scaler.transform(target.reshape(-1, 1)).flatten()

        weights = self._calculate_weights(target)

        df.loc[mask, self.sample_weight_column] = weights
        return data.copy_with(df)

    def _calculate_weights(self, target: np.ndarray) -> np.ndarray:
        """Calculate sample weights based on the configured method.

        Args:
            target: Array of target values to compute weights from.

        Returns:
            Array of weights in range [weight_floor, 1.0].

        Raises:
            ValueError: If an unknown weighting method is configured.
        """
        match self.method:
            case "exponential":
                return exponential_sample_weight(
                    x=target,
                    scale_percentile=self.weight_scale_percentile,
                    exponent=self.weight_exponent,
                    floor=self.weight_floor,
                )
            case "inverse_frequency":
                return inverse_frequency_sample_weight(
                    x=target,
                    n_bins=self.n_bins,
                    dampening_exponent=self.dampening_exponent,
                    floor=self.weight_floor,
                )
            case _:
                msg = f"Unknown weighting method: {self.method}"
                raise ValueError(msg)

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
    if x.size == 0:
        return np.empty_like(x)

    scaling_value = np.percentile(np.abs(x), scale_percentile)
    if scaling_value == 0:
        return np.full_like(x, fill_value=1.0)

    x_scaled = np.abs(x / scaling_value)
    x_weighted = np.abs(x_scaled) ** exponent
    return np.clip(x_weighted, a_min=floor, a_max=1.0)


def inverse_frequency_sample_weight(
    x: np.ndarray,
    n_bins: int = 50,
    dampening_exponent: float = 0.5,
    floor: float = 0.1,
) -> np.ndarray:
    """Calculate sample weights based on inverse frequency using histogram binning.

    Values that occur more frequently receive lower weights, while values that
    occur less frequently (rare samples) receive higher weights.

    Args:
        x: Array of target values to compute weights from.
        n_bins: Number of equal-width histogram bins for frequency estimation.
        dampening_exponent: Exponent in [0, 1] applied to inverse frequency ratio.
            Lower values compress the weight range, reducing impact of very rare
            samples. Use 1.0 for linear (no dampening), 0.0 for uniform weights.
        floor: Minimum weight value. Ensures all samples contribute to training.

    Returns:
        Array of weights in range [floor, 1.0] with same shape as input.
    """
    if x.size == 0:
        return np.empty_like(x)

    # Compute histogram with equal-width bins
    counts, bin_edges = np.histogram(x, bins=n_bins)

    # Assign each value to a bin
    bin_indices = np.digitize(x, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Get the count for each sample's bin
    sample_counts = counts[bin_indices]

    # Calculate inverse frequency ratio
    max_count = counts.max()
    inverse_freq_ratio = max_count / sample_counts

    # Apply dampening exponent to compress the range
    dampened_ratio = np.power(inverse_freq_ratio, dampening_exponent)

    # Normalize to [floor, 1.0]
    weights = dampened_ratio / dampened_ratio.max()
    return weights * (1.0 - floor) + floor
