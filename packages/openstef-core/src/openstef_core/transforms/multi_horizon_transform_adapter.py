# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Multi-horizon transform adapter for time series data.

Provides utilities for applying single-horizon transforms to multi-horizon
datasets, including concatenation and adaptation mechanisms.
"""

from typing import Self, override

import pandas as pd

from openstef_core.base_model import BaseModel
from openstef_core.datasets import MultiHorizonTimeSeriesDataset, TimeSeriesDataset
from openstef_core.mixins import State
from openstef_core.transforms.dataset_transforms import MultiHorizonTimeSeriesTransform, TimeSeriesTransform
from openstef_core.types import LeadTime


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


class MultiHorizonTransformAdapter(BaseModel, MultiHorizonTimeSeriesTransform):
    """Adapter to apply TimeSeriesTransform to multi-horizon datasets.

    This adapter allows single-horizon transforms to work with multi-horizon
    datasets by applying the transform to each horizon independently.
    The transform is fitted on the concatenated data from all horizons.

    Example:
        Using an adapter with a scaling transform:

        >>> from openstef_models.transforms.general.scaler_transform import ScalerTransform
        >>> scale_transform = ScalerTransform()
        >>> adapter = MultiHorizonTransformAdapter(time_series_transform=scale_transform)
        >>> # adapter can now be used with MultiHorizonTimeSeriesDataset
    """

    time_series_transform: TimeSeriesTransform

    @override
    def to_state(self) -> State:
        """Serialize the adapter state.

        Returns:
            Serialized state of the underlying time series transform.
        """
        return self.time_series_transform.to_state()

    @override
    def from_state(self, state: State) -> Self:
        return self.__class__(time_series_transform=self.time_series_transform.from_state(state))

    @property
    @override
    def is_fitted(self) -> bool:
        return self.time_series_transform.is_fitted

    @override
    def fit(self, data: MultiHorizonTimeSeriesDataset) -> None:
        """Fit the transform on concatenated multi-horizon data.

        Args:
            data: Multi-horizon dataset to fit on.
        """
        flat_data: TimeSeriesDataset = concat_horizon_datasets_rowwise(horizon_datasets=data)
        self.time_series_transform.fit(flat_data)

    @override
    def transform(self, data: MultiHorizonTimeSeriesDataset) -> MultiHorizonTimeSeriesDataset:
        return {horizon: self.time_series_transform.transform(dataset) for horizon, dataset in data.items()}


__all__ = ["MultiHorizonTransformAdapter", "concat_horizon_datasets_rowwise"]
