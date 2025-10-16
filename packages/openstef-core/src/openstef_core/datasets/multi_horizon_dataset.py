# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Multi-horizon time series dataset management.

Provides the MultiHorizon class for managing collections of time series datasets,
each associated with a specific forecast horizon. This enables efficient handling
of forecast data across different time horizons while maintaining consistency in
sample intervals. Different horizons may have different feature sets.
"""

import functools
import json
import operator
from collections import UserDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import override

import pandas as pd

from openstef_core.datasets.mixins import StoreableDatasetMixin, TimeSeriesMixin
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.validation import validate_same_sample_intervals
from openstef_core.datasets.versioned_timeseries import VersionedTimeSeriesDataset
from openstef_core.exceptions import TimeSeriesValidationError
from openstef_core.types import LeadTime
from openstef_core.utils import timedelta_from_isoformat, timedelta_to_isoformat
from openstef_core.utils.pandas import combine_timeseries_indexes


class MultiHorizon[T: TimeSeriesDataset](UserDict[LeadTime, T], TimeSeriesMixin, StoreableDatasetMixin):
    """A dictionary mapping forecast horizons to time series datasets.

    This class represents a collection of time series datasets, each associated
    with a specific forecast horizon. It extends the standard dictionary to
    provide additional type safety and clarity when working with multiple
    horizon-specific datasets.

    Attributes:
        keys: Forecast horizons (LeadTime) for which datasets are available.
        values: Corresponding time series datasets for each horizon.

    Example:
        Create a MultiHorizon dataset with two horizons:

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.types import LeadTime
        >>> from openstef_core.datasets import TimeSeriesDataset, MultiHorizon
        >>> # Create sample datasets for two horizons
        >>> data_1h = pd.DataFrame({
        ...     'load': [100, 110],
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='1h'))
        >>> dataset_1h = TimeSeriesDataset(data_1h, sample_interval=timedelta(hours=1))
        >>> data_24h = pd.DataFrame({
        ...     'load': [120, 130],
        ... }, index=pd.date_range('2025-01-01', periods=2, freq='1h'))
        >>> dataset_24h = TimeSeriesDataset(data_24h, sample_interval=timedelta(hours=1))
        >>> # Create MultiHorizon mapping
        >>> horizons = MultiHorizon({
        ...     LeadTime.from_string("PT1H"): dataset_1h,
        ...     LeadTime.from_string("P1D"): dataset_24h,
        ... })
        >>> len(horizons)
        2
        >>> list(horizons.keys())
        [LeadTime('PT1H'), LeadTime('P1D')]
    """

    def __init__(self, initial_data: dict[LeadTime, T]) -> None:
        """Initialize a MultiHorizon dataset with the given horizon-dataset mapping.

        Args:
            initial_data: Dictionary mapping LeadTime horizons to TimeSeriesDataset instances.

        Raises:
            TimeSeriesValidationError: If the initial data is empty or if datasets have
                inconsistent sample intervals.

        Note:
            Different horizons may have different feature sets (typically shorter
            lead times have more features than longer ones).
        """
        if len(initial_data) == 0:
            raise TimeSeriesValidationError("Initial data dictionary is empty.")

        self.sample_interval = validate_same_sample_intervals(datasets=initial_data.values())
        self.feature_names = list(
            functools.reduce(operator.ior, [set(d.feature_names) for d in initial_data.values()], set[str]())
        )
        self.index = combine_timeseries_indexes(indexes=[dataset.index for dataset in initial_data.values()])

        super().__init__(initial_data)

    def horizons(self) -> Sequence[LeadTime]:
        """Get the list of forecast horizons available in the MultiHorizon dataset.

        Returns:
            List of LeadTime objects representing the forecast horizons.
        """
        return list(self.keys())

    def map_horizons[U: TimeSeriesDataset](self, func: Callable[[T], U]) -> "MultiHorizon[U]":
        """Apply a function to each dataset in the MultiHorizon collection.

        Args:
            func: A callable that takes a TimeSeriesDataset and returns a new dataset.

        Returns:
            A new MultiHorizon instance with the transformed datasets.
        """
        return MultiHorizon({k: func(v) for k, v in self.items()})

    def to_versioned(self) -> VersionedTimeSeriesDataset:
        """Convert the multi-horizon dataset to a versioned time series dataset.

        Transforms horizon-specific datasets into a single versioned dataset where each
        forecast is tagged with its available_at timestamp based on the forecast horizon.
        The available_at time is calculated by subtracting the horizon from the forecast
        timestamp.

        Returns:
            VersionedTimeSeriesDataset containing all forecasts with their availability times.
        """
        return VersionedTimeSeriesDataset.from_dataframe(
            data=pd.concat(
                [
                    dataset.data.assign(available_at=dataset.data.index.to_series() - horizon.value).reset_index(
                        names="timestamp"
                    )
                    for horizon, dataset in self.items()
                ],
                axis=0,
            ),
            sample_interval=self.sample_interval,
            timestamp_column="timestamp",
            available_at_column="available_at",
        )

    @override
    def to_parquet(self, path: Path) -> None:
        combined_data = pd.concat(
            [dataset.data.assign(horizon=str(horizon)) for horizon, dataset in self.items()], axis=0
        )
        combined_data.attrs["sample_interval"] = timedelta_to_isoformat(self.sample_interval)
        combined_data.attrs["structure"] = json.dumps({
            "horizons": [{"horizon": str(h), "columns": d.data.columns.tolist()} for h, d in self.items()]
        })
        combined_data.to_parquet(path)

    @override
    @classmethod
    def read_parquet(cls, path: Path) -> "MultiHorizon[TimeSeriesDataset]":
        combined_data: pd.DataFrame = pd.read_parquet(path)  # type: ignore
        sample_interval = timedelta_from_isoformat(combined_data.attrs.get("sample_interval", "PT1H"))
        structure = json.loads(combined_data.attrs.get("structure", "{}"))

        horizon_data: dict[LeadTime, TimeSeriesDataset] = {}
        for info in structure.get("horizons", []):
            horizon = LeadTime.from_string(info["horizon"])
            columns: list[str] = info["columns"]
            group = combined_data.loc[combined_data.horizon == info["horizon"], columns]
            horizon_data[horizon] = TimeSeriesDataset(data=group, sample_interval=sample_interval)

        return MultiHorizon(horizon_data)


__all__ = ["MultiHorizon"]
