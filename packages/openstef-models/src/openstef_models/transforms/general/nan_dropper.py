# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform for dropping rows containing NaN values.

This module provides functionality to drop rows containing NaN values in selected
columns, useful for data cleaning and ensuring complete cases for model training.
"""

import logging
from typing import override

from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.transforms import TimeSeriesTransform
from openstef_models.utils.feature_selection import FeatureSelection


class NaNDropper(BaseConfig, TimeSeriesTransform):
    """Transform that drops rows containing NaN values in selected columns.

    This transform removes any row that has at least one NaN value in the
    specified columns. It operates statelessly - no fitting is required.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.general import NaNDropper
        >>>
        >>> # Create sample dataset with NaN values
        >>> data = pd.DataFrame({
        ...     'load': [100.0, np.nan, 110.0, 130.0],
        ...     'temperature': [20.0, 22.0, np.nan, 23.0],
        ...     'humidity': [60.0, 65.0, 70.0, 75.0]
        ... }, index=pd.date_range('2025-01-01', periods=4, freq='1h'))
        >>> dataset = TimeSeriesDataset(data, timedelta(hours=1))
        >>>
        >>> # Drop rows with NaN in load or temperature
        >>> dropper = NaNDropper(selection=FeatureSelection(include=['load', 'temperature']))
        >>> transformed = dropper.transform(dataset)
        >>> len(transformed.data)
        2
        >>> transformed.data['load'].tolist()
        [100.0, 130.0]

    """

    selection: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description="Features to check for NaN values. Rows with NaN in any selected column are dropped.",
    )
    warn_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Log a warning if the fraction of dropped rows exceeds this threshold (0.0 to 1.0).",
    )

    _logger: logging.Logger = PrivateAttr(default_factory=lambda: logging.getLogger(__name__))

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        features = self.selection.resolve(data.feature_names)
        original_row_count = len(data.data)

        # Drop rows containing NaN in selected columns
        transformed_data = data.data.dropna(subset=features)  # pyright: ignore[reportUnknownMemberType]
        dropped_count = original_row_count - len(transformed_data)

        # Log warning if substantial percentage of rows was dropped
        if original_row_count > 0 and dropped_count / original_row_count > self.warn_threshold:
            self._logger.warning(
                "NaNDropper dropped %d of %d rows (%.1f%%) due to NaN values in columns %s",
                dropped_count,
                original_row_count,
                dropped_count / original_row_count * 100,
                features,
            )

        return data.copy_with(data=transformed_data, is_sorted=True)

    @override
    def features_added(self) -> list[str]:
        return []
