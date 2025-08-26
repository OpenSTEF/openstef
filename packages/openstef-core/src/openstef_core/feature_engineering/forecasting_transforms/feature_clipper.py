# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for clipping feature values to observed ranges.

This module provides functionality to clip feature values to their observed
minimum and maximum ranges during training, preventing out-of-range values
during inference and improving model robustness.
"""

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


class FeatureClipper(TimeSeriesTransform):
    """Transform that clips specified features to their observed min and max values.

    This transform learns the minimum and maximum values of specified features
    during the fit phase and clips any values outside this range during the transform phase.

    Example:
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_core.feature_engineering.forecasting_transforms.feature_clipper import FeatureClipper
        >>>
        >>> # Create sample training dataset
        >>> training_data = pd.DataFrame({
        ...     'load': [100, 120, 110, 130, 125],
        ...     'temperature': [20, 22, 21, 23, 24]
        ... }, index=pd.date_range('2025-01-01', periods=5, freq='1h'))
        >>> training_dataset = TimeSeriesDataset(training_data, timedelta(hours=1))
        >>> test_data = pd.DataFrame({
        ...     'load': [90, 140, 115],
        ...     'temperature': [19, 25, 22]
        ... }, index=pd.date_range('2025-01-06', periods=3,
        ... freq='1h'))
        >>> test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))
        >>> # Initialize and apply transform
        >>> clipper = FeatureClipper(column_names=['load', 'temperature'])
        >>> clipper.fit(training_dataset)
        >>> transformed_dataset = clipper.transform(test_dataset)
        >>> transformed_dataset.data['load'].tolist()
        [100, 130, 115]
        >>> transformed_dataset.data['temperature'].tolist()
        [20, 24, 22]

    """

    def __init__(self, column_names: list[str]) -> None:
        """Initialize the FeatureClipper.

        Initializes with specified column names and a dictionary containing
        the feature range.
        """
        self.column_names: list[str] = column_names
        self.feature_ranges: dict[str, tuple[float, float]] = {}

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fit the transform to the data by learning the min and max values.

        Args:
            data: Time series dataset.
        """
        for col in self.column_names:
            if col in data.data.columns:
                self.feature_ranges[col] = (data.data[col].min(), data.data[col].max())

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform the input time series data by clipping specified features to their learned min and max values.

        Args:
            data: Time series dataset.

        Returns:
            Transformed time series dataset with clipped features.
        """
        transformed_data = data.data.copy()

        for col in self.feature_ranges:
            if col in transformed_data.columns:
                min_val, max_val = self.feature_ranges[col]
                transformed_data[col] = transformed_data[col].clip(lower=min_val, upper=max_val)

        return TimeSeriesDataset(data=transformed_data, sample_interval=data.sample_interval)
