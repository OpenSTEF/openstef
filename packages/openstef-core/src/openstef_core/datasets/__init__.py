# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Time series datasets and versioned data access.

This module provides core data structures for handling time series data in OpenSTEF forecasting.
It includes both simple time series datasets and versioned datasets that track data availability
over time, enabling realistic backtesting and training and forecasting.

The module supports:
    - Regular time series with consistent sampling intervals
    - Versioned time series that track when data became available
    - Data transformations and validation utilities
    - Feature concatenation and horizon restriction operations
"""

from openstef_core.datasets import mixins
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_accessors import VersionedTimeSeriesAccessors
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeseriesDataset

__all__ = [
    "TimeSeriesDataset",
    "VersionedTimeSeriesAccessors",
    "VersionedTimeseriesDataset",
    "mixins",
]
