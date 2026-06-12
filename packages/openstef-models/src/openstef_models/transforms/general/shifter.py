# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for shifting features to align aggregation intervals.

This module provides functionality to shift time series features that are aggregated
over a different interval than the target variable, correcting the phase misalignment
by shifting and linearly interpolating back onto the original time grid.
"""

from datetime import timedelta
from typing import Any, override

import numpy as np
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class Shifter(BaseConfig, TimeSeriesTransform):
    """Transform that shifts features to align their aggregation interval with the target.

    When source features are aggregated over a different interval than the target variable,
    their timestamps represent a different center point in time. This transform corrects
    the phase misalignment by shifting the source features and linearly interpolating
    back onto the original time grid.

    The shift is computed as::

        shift = source_aggregation_period / 2 - target_aggregation_period / 2

    Timestamps are assumed to be at the end of the aggregation interval.
    For example, a timestamp of 12:00 with a 60-minute aggregation period represents
    the average over [11:00, 12:00], centered at 11:30. For instantaneous features
    or target, use an aggregation period of zero.

    Example: Aligning hourly radiation with 15-minute load
        Hourly radiation (source_aggregation_period=60 min) has its center 30 min
        before the timestamp, while 15-minute load (target_aggregation_period=15 min)
        has its center 7.5 min before the timestamp. The required backward shift
        for radiation is 22.5 minutes.

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import Shifter
        >>> from openstef_models.utils.feature_selection import FeatureSelection
        >>>
        >>> # Hourly radiation interpolated onto a 15-minute grid
        >>> index = pd.date_range('2025-01-01', periods=8, freq='15min')
        >>> data = pd.DataFrame({
        ...     'load': range(8),
        ...     'radiation': [200, 220, 240, 260, 280, 300, 320, 340],
        ... }, index=index)
        >>> dataset = TimeSeriesDataset(data, timedelta(minutes=15))
        >>>
        >>> shifter = Shifter(
        ...     selection=FeatureSelection(include=['radiation']),
        ...     source_aggregation_period=timedelta(minutes=60),
        ...     target_aggregation_period=timedelta(minutes=15),
        ...     fill_edges=True,
        ... )
        >>> result = shifter.transform(dataset)
        >>> result.data['radiation'].tolist()
        [230.0, 250.0, 270.0, 290.0, 310.0, 330.0, 340.0, 340.0]
    """

    selection: FeatureSelection = Field(
        default=FeatureSelection.NONE,
        description="Features to shift.",
    )
    source_aggregation_period: timedelta = Field(
        default=timedelta(minutes=60),
        description="Aggregation period of the source features.",
    )
    target_aggregation_period: timedelta = Field(
        default=timedelta(minutes=15),
        description="Aggregation period of the target variable.",
    )
    fill_edges: bool = Field(
        default=False,
        description=(
            "Whether to fill NaN at the edges introduced by the shift "
            "with the original (un-shifted) boundary value of each feature."
        ),
    )

    _shift: timedelta = PrivateAttr()

    @override
    def model_post_init(self, context: Any) -> None:
        self._shift = self.source_aggregation_period / 2 - self.target_aggregation_period / 2

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if self._shift == timedelta(0):
            return data

        features = self.selection.resolve(data.feature_names)
        transformed_data = data.data.copy()

        original_index = data.index
        shifted_index = original_index - self._shift
        combined_index = original_index.union(shifted_index)

        feature_data = transformed_data[features]

        # Place values on the shifted time axis, interpolate back onto the original grid
        shifted_df = feature_data.set_axis(shifted_index)
        combined_df = shifted_df.reindex(combined_index)

        limit_area = None if self.fill_edges else "inside"
        realigned = combined_df.interpolate(method="time", limit_direction="both", limit_area=limit_area)
        realigned = realigned.reindex(original_index)

        # Restore pre-existing NaN at their shifted positions (nearest-neighbor mapping)
        nan_mask_shifted = feature_data.isna().set_axis(shifted_index)
        realigned[nan_mask_shifted.reindex(original_index, method="nearest")] = np.nan

        transformed_data[features] = realigned

        return data.copy_with(data=transformed_data, is_sorted=True)

    @override
    def features_added(self) -> list[str]:
        return []
