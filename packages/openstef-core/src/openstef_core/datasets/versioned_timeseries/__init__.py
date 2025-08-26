# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Versioned time series datasets for tracking forecast evolution over time.

Provides data structures and utilities for managing time series data with multiple
forecast versions, enabling analysis of how predictions change as new information
becomes available and supporting realistic backtesting scenarios.
"""

from openstef_core.datasets.versioned_timeseries.accessors import concat_featurewise, restrict_horizon
from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.filters import (
    filter_by_available_at,
    filter_by_latest_lead_time,
    filter_by_lead_time,
)

__all__ = [
    "VersionedTimeSeriesDataset",
    "concat_featurewise",
    "filter_by_available_at",
    "filter_by_latest_lead_time",
    "filter_by_lead_time",
    "restrict_horizon",
]
