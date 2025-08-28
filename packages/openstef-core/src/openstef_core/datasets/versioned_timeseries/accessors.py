# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Compatibility layer for versioned time series accessors.

This module provides placeholders for accessor classes that were used
by the old implementation. These will redirect to the new V4 implementation.
"""

from datetime import datetime

import pandas as pd

from openstef_core.datasets.mixins import VersionedTimeSeriesMixin


class RestrictedHorizonVersionedTimeSeries:
    """Compatibility wrapper for horizon-restricted access.

    This is a placeholder that maintains the interface of the old implementation
    while using the new V4 classes underneath.
    """

    def __init__(self, dataset: VersionedTimeSeriesMixin, horizon: datetime) -> None:
        """Initialize with dataset and horizon.

        Args:
            dataset: The underlying dataset
            horizon: The horizon restriction time
        """
        self.dataset = dataset
        self.horizon = horizon

    @property
    def feature_names(self) -> list[str]:
        """Get feature names from underlying dataset."""
        return self.dataset.feature_names

    def get_window(
        self, start: datetime, end: datetime, available_before: datetime | None = None
    ) -> pd.DataFrame:
        """Get data window with horizon restriction.

        Returns:
            DataFrame with data from the specified window.
        """
        raise NotImplementedError("get_window is not implemented in the compatibility wrapper.")



def restrict_horizon(dataset: VersionedTimeSeriesMixin, horizon: datetime) -> RestrictedHorizonVersionedTimeSeries:
    """Restrict dataset to horizon.
    
    Returns:
        Wrapped dataset with horizon restriction.
    """
    return RestrictedHorizonVersionedTimeSeries(dataset, horizon)


__all__ = [
    "RestrictedHorizonVersionedTimeSeries",
    "restrict_horizon",
]
