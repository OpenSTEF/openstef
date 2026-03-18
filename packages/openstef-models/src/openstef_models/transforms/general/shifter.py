# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for shifting features to align averaging intervals.

This module provides functionality to shift time series features that are averaged
over a different interval than the target variable, correcting the phase misalignment
by shifting and linearly interpolating back onto the original time grid.
"""

from datetime import timedelta
from typing import Any, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class Shifter(BaseConfig, TimeSeriesTransform):
    """Transform that shifts features to align their averaging interval with the target.

    When source features are averaged over a different interval than the target variable,
    their timestamps represent a different center point in time. This transform corrects
    the phase misalignment by shifting the source features and linearly interpolating
    back onto the original time grid.

    The shift is computed as::

        shift = source_averaging_period / 2 - target_averaging_period / 2

    Timestamps are assumed to be at the end of the averaging interval.
    For example, a timestamp of 12:00 with a 60-minute averaging period represents
    the average over [11:00, 12:00], centered at 11:30. For instantaneous features
    or target, use an averaging period of zero.

    Example: Aligning hourly radiation with 15-minute load
        Hourly radiation (source_averaging_period=60 min) has its center 30 min
        before the timestamp, while 15-minute load (target_averaging_period=15 min)
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
        ...     source_averaging_period=timedelta(minutes=60),
        ...     target_averaging_period=timedelta(minutes=15),
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
    source_averaging_period: timedelta = Field(
        default=timedelta(minutes=60),
        description="Averaging period of the source features.",
    )
    target_averaging_period: timedelta = Field(
        default=timedelta(minutes=15),
        description="Averaging period of the target variable.",
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
        self._shift = self.source_averaging_period / 2 - self.target_averaging_period / 2

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        if self._shift == timedelta(0):
            return data

        features = self.selection.resolve(data.feature_names)
        transformed_data = data.data.copy()

        original_index = cast(pd.DatetimeIndex, data.data.index)
        shifted_index = original_index - self._shift
        combined_index = cast(pd.DatetimeIndex, original_index.union(shifted_index))

        feature_data = transformed_data[features]

        # Place values on the shifted time axis, interpolate back onto the original grid
        shifted_df = feature_data.set_axis(shifted_index)  # pyright: ignore[reportUnknownMemberType]
        realigned = shifted_df.reindex(combined_index).interpolate(method="time").reindex(original_index)

        # Restore pre-existing NaN at their shifted positions (nearest-neighbor mapping)
        nan_mask_shifted = feature_data.isna().set_axis(shifted_index)  # pyright: ignore[reportUnknownMemberType]
        realigned[nan_mask_shifted.reindex(original_index, method="nearest")] = np.nan

        # Handle timestamps outside the range covered by the shifted data
        outside_mask = (original_index < shifted_index.min()) | (original_index > shifted_index.max())  # pyright: ignore[reportUnknownMemberType]
        if self.fill_edges:
            edge = feature_data.iloc[[-1]] if self._shift > timedelta(0) else feature_data.iloc[[0]]
            realigned.loc[outside_mask] = edge.to_numpy()
        else:
            realigned.loc[outside_mask] = np.nan

        transformed_data[features] = realigned

        return data.copy_with(data=transformed_data, is_sorted=True)

    @override
    def features_added(self) -> list[str]:
        return []
