# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Quantile forecast ordering correction.

Ensures quantile forecasts maintain monotonic ordering (e.g., q10 ≤ q50 ≤ q90)
by sorting predictions across quantile columns. This fixes violations that can
occur when quantiles are predicted independently.
"""

from typing import Self, override

import numpy as np
import pandas as pd

from openstef_core.datasets import ForecastDataset
from openstef_core.mixins import State, Transform


class QuantileSorter(Transform[ForecastDataset, ForecastDataset]):
    """Sort quantile forecasts to enforce monotonic ordering.

    Probabilistic forecasts should have higher quantiles predict higher values
    (e.g., the 90th percentile should be greater than the 10th percentile). When
    quantiles are predicted independently, this property can be violated. This
    transform corrects violations by sorting quantile predictions for each time step.

    The transform is stateless and requires no fitting.

    Example:
        Basic usage with a forecast dataset:

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import ForecastDataset
        >>> # Create sample data with unsorted quantiles
        >>> data = pd.DataFrame({
        ...     'quantile_P10': [1.0, 2.0, 3.0],
        ...     'quantile_P50': [0.5, 1.5, 2.5],  # Violates ordering
        ...     'quantile_P90': [2.0, 3.0, 4.0]
        ... }, index=pd.date_range('2025-01-01', periods=3, freq='h'))
        >>> dataset = ForecastDataset(
        ...     data=data,
        ...     sample_interval=timedelta(hours=1)
        ... )
        >>> sorter = QuantileSorter()
        >>> sorted_dataset = sorter.transform(dataset)
        >>> # Now quantile_P10 <= quantile_P50 <= quantile_P90 for each time step
        >>> sorted_dataset.data.iloc[0].values.tolist()
        [0.5, 1.0, 2.0]
    """

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: ForecastDataset) -> None:
        pass  # noop - stateless transform

    @override
    def transform(self, data: ForecastDataset) -> ForecastDataset:
        quantile_columns = [quantile.format() for quantile in sorted(data.quantiles)]
        sorted_data = pd.DataFrame(
            data=np.sort(data.data[quantile_columns].values, axis=1),
            index=data.data.index,
            columns=quantile_columns,
        )

        return ForecastDataset(
            data=sorted_data,
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
            is_sorted=True,
        )

    @override
    def to_state(self) -> State:
        return None

    @override
    def from_state(self, state: State) -> Self:
        return self


__all__ = ["QuantileSorter"]
