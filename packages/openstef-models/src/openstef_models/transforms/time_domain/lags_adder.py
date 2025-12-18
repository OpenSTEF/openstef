# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Lag feature generation for time series forecasting.

Creates lagged versions of target variables to capture temporal patterns.
Supports multiple strategies: trivial lags (minute/day-based), custom lags,
and autocorrelation-based lags for adaptive feature engineering.
"""

import logging
import math
from datetime import timedelta
from typing import Any, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset, validate_horizons_present
from openstef_core.transforms import TimeSeriesTransform
from openstef_core.types import LeadTime
from openstef_core.utils import timedelta_to_isoformat

logger = logging.getLogger(__name__)


class LagsAdder(BaseConfig, TimeSeriesTransform):
    """Transform that adds lag features to time series data.

    Creates lagged copies of the target variable to capture temporal dependencies.
    Handles both single-horizon and multi-horizon (versioned) datasets, ensuring
    lags are valid for each forecast horizon.

    Invariants:
        - fit() must be called before transform() when add_autocorr_lags=True
        - Lags are only added if they fall within the available history window
        - For each horizon, only lags >= horizon are included (prevents data leakage)
        - All lag features use consistent naming: {target_column}_lag_{duration}
    """

    history_available: timedelta = Field(
        description="Duration for which historical data is available.",
    )
    horizons: list[LeadTime] = Field(
        description="List of forecast horizons to create lag features for.",
        min_length=1,
    )
    custom_lags: list[timedelta] | None = Field(
        default=None,
        description="Explicit list of lag durations to create. If None, lags will be inferred based on horizons.",
    )
    target_column: str = Field(
        default="load",
        description="The name of the target feature to create lag features for.",
    )
    max_day_lags: int = Field(
        default=14,
        description="Maximum number of days to look back for day-based lags. "
        "Default is 14 days (two weekly cycles), typical for energy forecasting.",
        ge=1,
    )
    add_trivial_lags: bool = Field(
        default=True,
        description="Whether to add trivial lag features (minute-based and day-based lags).",
    )
    add_autocorr_lags: bool = Field(
        default=False,
        description="Whether to add autocorrelation-based lag features.",
    )

    _lags: list[timedelta] = PrivateAttr(default_factory=list[timedelta])
    _horizon_lags: dict[LeadTime, list[timedelta]] = PrivateAttr(default_factory=dict[LeadTime, list[timedelta]])
    _is_fitted: bool = PrivateAttr(default=False)

    @property
    def horizon_lags(self) -> dict[LeadTime, list[timedelta]]:
        """Mapping of forecast horizons to their valid lag features."""
        return self._horizon_lags

    @property
    def lags(self) -> list[timedelta]:
        """All lag durations configured for this transform, sorted descending."""
        return self._lags

    @property
    def max_horizon(self) -> timedelta:
        """Longest forecast horizon to determine minimum lag requirements."""
        return max(horizon.value for horizon in self.horizons)

    def _add_lags(self, new_lags: list[timedelta]) -> None:
        self._lags = sorted(set(self._lags + new_lags), reverse=True)
        self._horizon_lags = {
            # Filter lags: must be far enough back (>= horizon) but within available history
            horizon: [lag for lag in self._lags if horizon.value <= lag <= self.history_available]
            for horizon in self.horizons
        }

    @override
    def model_post_init(self, context: Any) -> None:
        lags: list[timedelta] = []

        # Add trivial lags (minute-based and day-based) if enabled
        if self.add_trivial_lags:
            lags.extend(generate_minute_lags(max_horizon=self.max_horizon))
            lags.extend(generate_day_lags(max_horizon=self.max_horizon, max_day_lags=self.max_day_lags))

        # Add explicit lags if provided
        if self.custom_lags is not None:
            lags.extend(self.custom_lags)

        # Update lags and compute valid horizon lags
        self._add_lags(lags)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted or not self.add_autocorr_lags

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        if not self.add_autocorr_lags:
            self._is_fitted = True
            return

        # Extract target series for autocorrelation analysis
        target_series = data.select_version().data[self.target_column]

        # Generate autocorrelation-based lags
        autocorr_lags = generate_autocorr_lags(
            signal=target_series,
            max_horizon=self.max_horizon,
        )

        # Update lags with autocorr lags
        self._add_lags(autocorr_lags)

        self._is_fitted = True

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        validate_horizons_present(data, self.horizons)

        # Copy the input data to add lag features to
        df = data.data.copy(deep=False)

        if len(self.horizons) == 1:
            # For non-versioned data: use single horizon's valid lags
            horizon = self.horizons[0]
            valid_lags = self._horizon_lags.get(horizon, [])

            for lag in valid_lags:
                feature_name = self._lag_feature(lag)
                df[feature_name] = df[self.target_column].shift(freq=lag)
        else:
            # For versioned data: add lags based on each point's horizon
            # Pre-create all feature columns with NaN
            all_possible_lags = sorted({lag for lags in self._horizon_lags.values() for lag in lags})
            for lag in all_possible_lags:
                feature_name = self._lag_feature(lag)
                df[feature_name] = np.nan

            # Fill in values where they're valid for each horizon
            for horizon, valid_lags in self._horizon_lags.items():
                # Get rows with this horizon
                horizon_mask = df[data.horizon_column] == horizon.value

                for lag in valid_lags:
                    feature_name = self._lag_feature(lag)
                    # Shift the target column for rows matching this horizon
                    df.loc[horizon_mask, feature_name] = df.loc[horizon_mask, self.target_column].shift(freq=lag)

        return data.copy_with(data=df, is_sorted=True)

    def _lag_feature(self, lag: timedelta) -> str:
        return f"{self.target_column}_lag_{timedelta_to_isoformat(lag)}"

    @override
    def features_added(self) -> list[str]:
        # Return all possible feature names from all lags
        return [self._lag_feature(lag) for lag in self._lags]


def generate_minute_lags(max_horizon: timedelta) -> list[timedelta]:
    """Generate minute-based lag features for short-term forecasting.

    Creates hourly lags (1-23 hours) and sub-hourly lags (15, 30, 45 minutes)
    that are valid for the given forecast horizon.

    Args:
        max_horizon: Maximum forecast horizon - only lags >= this will be included.

    Returns:
        List of timedeltas representing valid minute-based lags, sorted descending.
    """
    # Create base set: hourly lags (1-23h) plus sub-hourly (15, 30, 45 min)
    hourly_lags = pd.timedelta_range(start="1h", end="23h", freq="1h")
    subhourly_lags = pd.to_timedelta([15, 30, 45], unit="min")  # pyright: ignore[reportUnknownMemberType]
    base_lags = pd.Index(hourly_lags).union(pd.Index(subhourly_lags))

    # Filter: only lags that are far enough in the past (>= forecast horizon)
    valid_lags = cast(pd.TimedeltaIndex, base_lags[base_lags >= max_horizon])  # type: ignore

    # Convert to Python timedelta list (no duplicates possible)
    return list(valid_lags.to_pytimedelta())


def generate_day_lags(max_horizon: timedelta, max_day_lags: int) -> list[timedelta]:
    """Generate day-based lag features for capturing daily and weekly patterns.

    Creates daily lags from the minimum required days up to the maximum allowed,
    useful for capturing day-of-week and weekly seasonality.

    Args:
        max_horizon: Maximum forecast horizon - only lags >= this will be included.
        max_day_lags: Maximum number of days to look back (typically 14 for two weekly cycles).

    Returns:
        List of timedeltas representing valid day-based lags, sorted descending.
        Empty list if minimum required days exceeds max_day_lags.
    """
    # Calculate minimum days needed to exceed the horizon (timedelta division returns float)
    min_days = math.ceil(max_horizon / timedelta(days=1))

    # If min_days exceeds the configured maximum, no day lags are possible
    if min_days > max_day_lags:
        return []

    # Generate day lags as a range from min_days to max_day_lags (inclusive)
    day_lags = pd.timedelta_range(start=pd.Timedelta(days=min_days), end=pd.Timedelta(days=max_day_lags), freq="1D")

    # Convert to Python timedelta list (no duplicates possible)
    return list(day_lags.to_pytimedelta())


def generate_autocorr_lags(
    signal: pd.Series,
    max_horizon: timedelta,
    height_threshold: float = 0.1,
    max_lag_hours: int = 4,
) -> list[timedelta]:
    """Generate lag features based on autocorrelation peaks in the time series.

    Analyzes the autocorrelation function of the input data to identify significant
    patterns. Peaks in the autocorrelation curve indicate time delays where the
    signal is similar to itself, suggesting useful lag features.

    Args:
        signal: Time series data to analyze (typically the target variable).
        max_horizon: Maximum forecast horizon - only lags >= this will be included.
        height_threshold: Minimum autocorrelation value to recognize as a peak.
            Higher values = fewer, more significant peaks. Default 0.1.
        max_lag_hours: Maximum lag time in hours to search for peaks. Default 4 hours.

    Returns:
        List of lag timedeltas corresponding to autocorrelation peaks, filtered by max_horizon.
        Returns empty list if scipy is not available, data is insufficient, or has no variance.

    """
    try:
        import scipy.signal  # noqa: PLC0415
    except ImportError:
        logger.warning("scipy not available, cannot generate autocorrelation-based lags")
        return []

    # Remove NaN values as autocorrelation handles them poorly
    # Cast to float array explicitly to help type checker
    clean_data = np.asarray(a=signal.dropna().values, dtype=np.float64)

    min_samples = 100
    if len(clean_data) < min_samples:
        logger.debug(
            "Insufficient data for autocorrelation analysis: %d samples (minimum %d required)",
            len(clean_data),
            min_samples,
        )
        return []

    # Compute autocorrelation using numpy
    mean_val = np.mean(clean_data)
    var_val = np.var(clean_data)

    if var_val == 0:  # Constant signal has no meaningful autocorrelation
        logger.debug("Signal has zero variance, cannot compute autocorrelation lags")
        return []

    centered = clean_data - mean_val
    # Full correlation, normalize, and take positive lags only
    corr = np.correlate(centered, centered, mode="full")  # type: ignore[assignment]
    corr = corr[len(corr) // 2 :] / var_val / len(centered)

    # Limit search to max_lag_hours (assuming 15-minute resolution)
    max_samples = int(max_lag_hours * 60 / 15)
    corr_subset = corr[: min(len(corr), max_samples)]

    # Find peaks in absolute autocorrelation (both positive and negative correlations matter)
    peaks_result = scipy.signal.find_peaks(np.abs(corr_subset), height=height_threshold)
    peaks = peaks_result[0]

    # Convert peak indices to lag times (assuming 15-minute data resolution)
    lag_minutes_array = peaks * 15

    # Convert to timedelta and filter by horizon
    lags = [timedelta(minutes=int(m)) for m in lag_minutes_array if timedelta(minutes=int(m)) >= max_horizon]

    logger.debug("Found %d autocorrelation-based lags for max_horizon=%s", len(lags), max_horizon)
    return sorted(lags, reverse=True)
