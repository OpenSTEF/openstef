# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform

NUM_DAYS_IN_YEAR = 365.25
NUM_DAYS_IN_WEEK = 7
NUM_MONTHS_IN_YEAR = 12


class CyclicFeatures(TimeSeriesTransform):
    """A transform for generating cyclic temporal features from time series data.

    This class converts temporal information (hours, days, months, seasons) into
    cyclic sine and cosine components that preserve the periodic nature of time.
    This encoding ensures that temporal boundaries (e.g., end of day/week/year)
    are properly connected, improving model performance for time-dependent patterns.

    The transform generates the following cyclic feature pairs:
        - Season: Based on day of year (365.25 day cycle)
        - Weekday: Based on day of week (7 day cycle)
        - Month: Based on month of year (12 month cycle)

    Each temporal component is encoded as both sine and cosine values to provide
    a complete 2D representation of the cyclic position.

    Example:
        Generate cyclic features for a time series:

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'load': [100, 120, 110]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(hours=1))
        >>> # Apply cyclic features
        >>> transform = CyclicFeatures()
        >>> transform.fit(dataset)
        >>> transformed = transform.transform(dataset)
        >>> 'season_sine' in transformed.feature_names
        True
        >>> 'day0fweek_cosine' in transformed.feature_names
        True

    Note:
        The transform must be fitted before it can be applied to transform data.
        Fitted features are stored and reused for consistent transformations.
    """

    cyclic_features: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _compute_sine(phase: pd.Index, period: float) -> np.ndarray:
        """Compute the sine component for cyclic features.

        Converts a phase index into sine values for cyclic encoding. The sine
        component represents one dimension of the 2D cyclic representation.

        Args:
            phase: Index values representing the position in the cycle
                (e.g., day of year, hour of day).
            period: The length of the complete cycle (e.g., 365.25 for yearly,
                24 for daily).

        Returns:
            Series with sine values ranging from -1 to 1, with the same index
            as the input phase.

        Note:
            Used in combination with cosine to create a complete 2D cyclic
            representation that preserves temporal continuity.
        """
        return np.sin(2 * np.pi * phase / period)

    @staticmethod
    def _compute_cosine(phase: pd.Index, period: float) -> np.ndarray:
        """Compute the cosine component for cyclic features.

        Converts a phase index into cosine values for cyclic encoding. The cosine
        component represents the second dimension of the 2D cyclic representation.

        Args:
            phase: Index values representing the position in the cycle
                (e.g., day of year, hour of day).
            period: The length of the complete cycle (e.g., 365.25 for yearly,
                24 for daily).

        Returns:
            Series with cosine values ranging from -1 to 1, with the same index
            as the input phase.

        Note:
            Used in combination with sine to create a complete 2D cyclic
            representation that preserves temporal continuity.
        """
        return np.cos(2 * np.pi * phase / period)

    def _compute_seasonal_feature(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Compute seasonal cyclic features based on day of year.

        Creates sine and cosine components representing the position within
        the yearly cycle. Uses 365.25 days to account for leap years.

        Args:
            data: Time series dataset with DatetimeIndex.

        Returns:
            Dictionary containing 'season_sine' and 'season_cosine' series
            representing the yearly seasonal cycle.

        Note:
            Day of year ranges from 1-366, providing smooth seasonal transitions
            that connect December 31st to January 1st.
        """
        return TimeSeriesDataset(
            data=pd.DataFrame(
                {
                    "season_sine": self._compute_sine(data.index.dayofyear, NUM_DAYS_IN_YEAR),
                    "season_cosine": self._compute_cosine(data.index.dayofyear, NUM_DAYS_IN_YEAR),
                }
            ),
            sample_interval=data.sample_interval
        )

    def _compute_weekday_feature(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Compute weekday cyclic features based on day of week.

        Creates sine and cosine components representing the position within
        the weekly cycle. Monday is 0, Sunday is 6.

        Args:
            data: Time series dataset with DatetimeIndex.

        Returns:
            Dictionary containing 'day0fweek_sine' and 'day0fweek_cosine' series
            representing the weekly cycle.

        Note:
            Ensures smooth transition from Sunday (6) back to Monday (0),
            preserving weekly patterns across week boundaries.
        """
        return TimeSeriesDataset(
            data=pd.DataFrame(
                {
                    "day0fweek_sine": self._compute_sine(data.index.day_of_week, NUM_DAYS_IN_WEEK),
                    "day0fweek_cosine": self._compute_cosine(data.index.day_of_week, NUM_DAYS_IN_WEEK),
                }
            ),
            sample_interval=data.sample_interval,
        )

    def _compute_monthly_feature(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Compute monthly cyclic features based on month of year.

        Creates sine and cosine components representing the position within
        the yearly monthly cycle. January is 1, December is 12.

        Args:
            data: Time series dataset with DatetimeIndex.

        Returns:
            Dictionary containing 'month_sine' and 'month_cosine' series
            representing the monthly cycle within a year.

        Note:
            Provides smooth transition from December (12) back to January (1),
            capturing annual monthly patterns and seasonal transitions.
        """
        return TimeSeriesDataset(
            data=pd.DataFrame(
                {
                    "month_sine": self._compute_sine(data.index.month, NUM_MONTHS_IN_YEAR),
                    "month_cosine": self._compute_cosine(data.index.month, NUM_MONTHS_IN_YEAR),
                }
            ),
            sample_interval=data.sample_interval,
        )

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the input time series data.

        Computes and stores all cyclic features based on the temporal information
        in the dataset's DatetimeIndex. The fitted features include seasonal,
        weekday, and monthly cyclic components.

        Args:
            data: Time series dataset to fit the transform on. Must have a
                DatetimeIndex for extracting temporal information.

        Note:
            This method must be called before transform() can be used. The fitted
            cyclic features are stored in the instance and reused for consistent
            transformations across different datasets with the same temporal structure.
        """
        self.cyclic_features = pd.concat(
            [
                self._compute_seasonal_feature(data),
                self._compute_weekday_feature(data),
                self._compute_monthly_feature(data),
            ],
            axis=1)

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data by adding cyclic features.

        Adds all fitted cyclic features to the dataset as new columns. The
        features include sine and cosine components for seasonal, weekday,
        and monthly cycles.

        Args:
            data: Time series dataset to transform. Should have the same
                temporal structure as the data used for fitting.

        Returns:
            The input dataset with added cyclic feature columns. The original
            data columns are preserved.

        Raises:
            AttributeError: If the transform has not been fitted yet (no
                cyclic_features stored).

        Example:
            Transform a dataset with fitted cyclic features:

            >>> # Assuming transform is already fitted
            >>> original_features = len(dataset.feature_names)
            >>> transformed = transform.transform(dataset)
            >>> len(transformed.feature_names) == original_features + 6
            True

        Note:
            The transform modifies the input dataset in-place by adding new
            feature columns. If existing features have the same names as
            cyclic features, they will be overwritten with a warning.
        """
        return data.ad
