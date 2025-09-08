# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Versioned time series datasets for realistic forecasting scenarios.

This module enables realistic backtesting and evaluation by modeling real-world
data availability constraints. Unlike traditional approaches that assume all
historical data is immediately available, versioned datasets track when each
data point actually became available for use.

This allows you to simulate what a forecast model would have known at any point
in time, enabling more accurate performance evaluation and preventing lookahead bias.

The package provides two key abstractions:
- Individual data parts that track availability timing
- Composite datasets that combine multiple data sources

Use these classes when you need to evaluate forecasting models under realistic
data availability conditions.
"""

from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.dataset_part import VersionedTimeSeriesPart

__all__ = [
    "VersionedTimeSeriesDataset",
    "VersionedTimeSeriesPart",
]
