# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for splitting versioned datasets by forecast horizons.

The main transform, HorizonSplitTransform, filters versioned data by lead time and
resolves timestamps to create clean time series datasets for each forecast horizon.
This enables efficient processing of multi-horizon forecasting models.
"""

from typing import Self, override

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import (
    MultiHorizon,
    TimeSeriesDataset,
    VersionedTimeSeriesDataset,
)
from openstef_core.mixins import State, Transform
from openstef_core.types import LeadTime


class HorizonSplitTransform(BaseConfig, Transform[VersionedTimeSeriesDataset, MultiHorizon[TimeSeriesDataset]]):
    """Transform that splits versioned datasets into horizon-specific time series.

    This transform is a key component in multi-horizon forecasting pipelines. It takes
    versioned time series data (containing forecast validity timestamps) and produces
    separate regular time series datasets for each specified forecast horizon.

    The transformation process:
    1. Filters data by each configured lead time (forecast horizon)
    2. Resolves timestamps to remove versioning complexity
    3. Returns clean time series datasets ready for standard processing

    Example:
        Basic usage with energy forecasting data:

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import VersionedTimeSeriesDataset
        >>> from openstef_core.types import LeadTime
        >>>
        >>> # Create sample versioned dataset
        >>> data = pd.DataFrame({
        ...     'timestamp': pd.date_range('2025-01-01', periods=24, freq='1h'),
        ...     'available_at': pd.date_range('2025-01-01', periods=24, freq='1h'),
        ...     'load': range(100, 124),
        ...     'temperature': [20 + i*0.5 for i in range(24)]
        ... })
        >>> versioned_dataset = VersionedTimeSeriesDataset.from_dataframe(data, sample_interval=timedelta(hours=1))
        >>>
        >>> # Configure horizons
        >>> horizons = [LeadTime.from_string("PT1H"), LeadTime.from_string("PT24H")]
        >>> splitter = HorizonSplitTransform(horizons=horizons)
        >>>
        >>> # Split into horizon-specific datasets
        >>> horizon_datasets = splitter.transform(versioned_dataset)
        >>> len(horizon_datasets)
        2
    """

    horizons: list[LeadTime] = Field(
        default_factory=lambda: [LeadTime.from_string("PT36H")],
        description="List of forecast horizons to prepare / split the dataset for.",
    )

    @override
    def to_state(self) -> State:
        return {"horizons": [str(horizon) for horizon in self.horizons]}

    @override
    def from_state(self, state: State) -> Self:
        return super().from_state(state)

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: VersionedTimeSeriesDataset) -> None:
        pass

    @override
    def transform(self, data: VersionedTimeSeriesDataset) -> MultiHorizon[TimeSeriesDataset]:
        """Split versioned dataset into horizon-specific time series datasets.

        For each configured horizon, filters the versioned data and resolves timestamps
        to create clean time series datasets suitable for standard processing.

        Args:
            data: Versioned time series dataset with forecast validity timestamps.

        Returns:
            Dictionary mapping each LeadTime to its corresponding TimeSeriesDataset.
        """
        return MultiHorizon({
            horizon: data.filter_by_lead_time(lead_time=horizon).select_version() for horizon in self.horizons
        })


__all__ = ["HorizonSplitTransform"]
