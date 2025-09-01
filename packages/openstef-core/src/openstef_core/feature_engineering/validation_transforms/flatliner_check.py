# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Flatliner check transform for time series datasets.

This module provides functionality for detecting flatliner patterns in time series load data.
A flatliner is defined as a period where the load remains constant for a specified duration, which can indicate sensor
malfunction, data transmission errors, or other anomalies in energy forecasting datasets.
"""

from datetime import timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform


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
    >>> transform.fit(dataset)
    >>> transform.flatliner_indicator == True
    True
    """

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
    _flatliner_indicator: bool = PrivateAttr(default=False)

    @property
    def flatliner_indicator(self) -> bool:
        """Indicates if a flatliner is currently detected after fitting."""
        return self._flatliner_indicator

    def __init__(self, **data: Any):
        """Initializes the FlatlinerCheckTransform with the given configuration."""
        super().__init__(**data)

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

    def fit(self, data: TimeSeriesDataset) -> None:
        """Fits the flatliner check by detecting ongoing flatliner patterns.

        This method checks for flatliner patterns in the 'load' column of the provided TimeSeriesDataset.

        Args:
            data: The dataset containing a DataFrame with a 'load' column to be checked for flatliner patterns.

        Raises:
            ValueError: If the input DataFrame does not contain a 'load' column.
        """
        if "load" not in data.data.columns:
            raise ValueError("The DataFrame must contain a 'load' column.")
        self._flatliner_indicator = self.detect_ongoing_flatliner(
            data=data.data["load"],
        )

    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:  # noqa: PLR6301
        """This method returns the input data unchanged.

        Args:
            data: The input time series dataset to be transformed.

        Returns:
            The unmodified input TimeSeriesDataset.
        """
        return data
