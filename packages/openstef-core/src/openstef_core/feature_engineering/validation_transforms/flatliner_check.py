# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Flatliner check transform for time series datasets.

This module provides functionality for detecting flatliner patterns in time series load data.
A flatliner is defined as a period where the load remains constant for a specified duration, which can indicate sensor
malfunction, data transmission errors, or other anomalies in energy forecasting datasets.
"""

from datetime import timedelta
from typing import cast, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform
from openstef_core.exceptions import FlatlinerDetectedError, MissingColumnsError


class FlatlinerCheckTransform(TimeSeriesTransform, BaseConfig):
    """Transformer to detect flatliner patterns in time series load data.

    A flatliner is a period where the load remains constant for a specified duration.
    This class can detect both zero and non-zero flatliners, depending on configuration.

    Example:
    >>> from datetime import timedelta
    >>> import numpy as np
    >>> import pandas as pd
    >>> from openstef_core.datasets import TimeSeriesDataset
    >>> from openstef_core.feature_engineering.validation_transforms.flatliner_check import (
    ...     FlatlinerCheckTransform,
    ... )
    >>> data = pd.DataFrame(
    ...     {
    ...         "load": [100, 110, 110, 110],
    ...     },
    ...     index=pd.date_range("2025-01-01", periods=4, freq="1h"),
    ... )
    >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
    >>> transform = FlatlinerCheckTransform(
    ...     flatliner_threshold_minutes=120,
    ...     detect_non_zero_flatliner=True,
    ...     relative_tolerance=1e-5
    ... )
    >>> try:
    ...     transform.fit(dataset)
    ... except FlatlinerDetectedError as e:
    ...     pass
    >>> transform.is_flatliner_detected == True
    True
    """

    load_column: str = Field(
        default="load",
        description="Name of the column to check for flatliners.",
    )
    flatliner_threshold_minutes: int = Field(
        default=1440,
        description="Number of minutes that the load has to be constant to detect a flatliner.",
    )
    detect_non_zero_flatliner: bool = Field(
        default=False,
        description="If True, flatliners are also detected on non-zero values (median of the load).",
    )
    absolute_tolerance: float = Field(
        default=0.0,
        description="The absolute tolerance for considering values as equal when detecting flatliners.",
    )
    relative_tolerance: float = Field(
        default=1e-5,
        description="The relative tolerance for considering values as equal when detecting flatliners.",
    )
    error_on_flatliner: bool = Field(
        default=True,
        description="If True, an error is raised when a flatliner is detected.",
    )
    check_on_transform: bool = Field(
        default=False,
        description="If True, flatliner detection also runs during transform() for new data validation.",
    )
    _is_flatliner_detected: bool | None = PrivateAttr(default=None)

    @property
    def is_flatliner_detected(self) -> bool | None:
        """Indicates if a flatliner is currently detected after fitting."""
        return self._is_flatliner_detected

    def detect_ongoing_flatliner(
        self,
        data: pd.Series,
    ) -> bool:
        """Detects if the latest measurements follow a flatliner pattern.

        The following equation is used to test whether two floats are equivalent:
        absolute(measurement - flatliner_value) <= (atol + rtol * absolute(flatliner_value))

        Args:
            data: A timeseries of measured load with a DatetimeIndex.
            duration_threshold_minutes: A flatliner is only detected if it exceeds the threshold duration.
            detect_non_zero_flatliner: If True, a flatliner is detected for non-zero values. If False,
                a flatliner is detected for zero values only.

        Returns:
            Boolean indicating whether or not there is a flatliner ongoing for the given data.
        """
        latest_measurement_time = cast(pd.Timestamp, data.last_valid_index())
        start_time = latest_measurement_time - timedelta(minutes=self.flatliner_threshold_minutes)
        latest_measurements = data[start_time:latest_measurement_time].dropna()

        flatliner_value = latest_measurements.median() if self.detect_non_zero_flatliner else 0

        flatline_condition = np.isclose(
            a=latest_measurements,
            b=flatliner_value,
            atol=self.absolute_tolerance,
            rtol=self.relative_tolerance,
        ).all()
        non_empty_condition = not latest_measurements.empty

        return bool(flatline_condition & non_empty_condition)

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        """Fits the flatliner check by detecting ongoing flatliner patterns.

        This method checks for flatliner patterns in the load column of the provided TimeSeriesDataset.

        Args:
            data: The dataset containing a DataFrame with a load column to be checked for flatliner patterns.

        Raises:
            ValueError: If the input DataFrame does not contain a load column.
            FlatlinerDetectedError: If a flatliner is detected and `error_on_flatliner` is set to True.
        """
        if self.load_column not in data.data.columns:
            raise MissingColumnsError([self.load_column])
        self._is_flatliner_detected = self.detect_ongoing_flatliner(
            data=data.data[self.load_column],
        )
        if self.error_on_flatliner and self._is_flatliner_detected:
            raise FlatlinerDetectedError("Flatliner detected in the provided load data.")

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Returns the input data unchanged, optionally checking for flatliners.

        This method can optionally run flatliner detection on new incoming data
        when `check_on_transform=True`, which is useful for real-time validation
        during forecasting.

        Args:
            data: The input time series dataset to be transformed.

        Returns:
            The unmodified input TimeSeriesDataset.

        Raises:
            ValueError: If the input DataFrame does not contain a load column and check_on_transform is True.
            FlatlinerDetectedError: If a flatliner is detected and `error_on_flatliner` is set to True.
        """
        if not self.check_on_transform:
            return data

        if self.load_column not in data.data.columns:
            raise MissingColumnsError([self.load_column])
        self._is_flatliner_detected = self.detect_ongoing_flatliner(
            data=data.data[self.load_column],
        )
        if self.error_on_flatliner and self._is_flatliner_detected:
            raise FlatlinerDetectedError("Flatliner detected in the provided load data.")

        return data
