# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for splitting versioned datasets by forecast horizons.

The main transform, HorizonSplitTransform, filters versioned data by lead time and
resolves timestamps to create clean time series datasets for each forecast horizon.
This enables efficient processing of multi-horizon forecasting models.
"""

import pandas as pd
from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.types import LeadTime


class HorizonSplitTransform(BaseConfig):
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

    def transform(self, dataset: VersionedTimeSeriesDataset) -> dict[LeadTime, TimeSeriesDataset]:
        """Split versioned dataset into horizon-specific time series datasets.

        For each configured horizon, filters the versioned data and resolves timestamps
        to create clean time series datasets suitable for standard processing.

        Args:
            dataset: Versioned time series dataset with forecast validity timestamps.

        Returns:
            Dictionary mapping each LeadTime to its corresponding TimeSeriesDataset.
        """
        return {horizon: dataset.filter_by_lead_time(lead_time=horizon).select_version() for horizon in self.horizons}


def concat_horizon_datasets_rowwise(horizon_datasets: dict[LeadTime, TimeSeriesDataset]) -> TimeSeriesDataset:
    """Concatenate multiple horizon datasets into a single dataset.

    This function takes a dictionary of horizon datasets, each corresponding to a specific lead time,
    and concatenates them row-wise into a single TimeSeriesDataset. The resulting dataset contains all
    the data from the input datasets, with the same sample interval as the first dataset in the input.

    Args:
        horizon_datasets: A dictionary where keys are LeadTime objects and values are TimeSeriesDataset instances.

    Returns:
        A single TimeSeriesDataset containing all rows from the input datasets.
    """
    return TimeSeriesDataset(
        data=pd.concat([horizon_data.data for horizon_data in horizon_datasets.values()]),
        sample_interval=horizon_datasets[next(iter(horizon_datasets))].sample_interval,
    )
