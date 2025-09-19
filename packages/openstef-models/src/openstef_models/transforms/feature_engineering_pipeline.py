# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Any, override

from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseModel
from openstef_core.datasets import (
    SelfTransform,
    TimeSeriesDataset,
    TimeSeriesTransform,
    Transform,
    VersionedTimeSeriesDataset,
)
from openstef_core.datasets.timeseries_dataset import MultiHorizonTimeSeriesDataset
from openstef_core.exceptions import TransformNotFittedError
from openstef_core.types import LeadTime
from openstef_models.transforms.horizon_split_transform import HorizonSplitTransform


class FeatureEngineeringPipeline(
    BaseModel, Transform[VersionedTimeSeriesDataset | TimeSeriesDataset, MultiHorizonTimeSeriesDataset]
):
    """Feature engineering pipeline for multi-horizon forecasting models.

    Orchestrates feature transformations in a two-stage process optimized for forecasting workflows:

    1. **Versioned transforms**: Applied to raw forecasting data that includes forecast validity timestamps,
       handling complex time logic like feature availability windows and forecast horizon dependencies.

    2. **Horizon transforms**: Applied to resolved time series data for each specific forecast horizon,
       performing standard feature engineering like scaling, aggregation, and encoding.

    This design separates time-aware transformations from standard feature engineering, ensuring
    efficient processing while maintaining forecast data integrity. The pipeline supports both
    versioned and unversioned (single horizon) datasets.

    Examples:
        **Example 1: Pipeline with versioned time series dataset**

        Creating and processing a versioned dataset for multi-horizon forecasting:

        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> from openstef_core.types import LeadTime
        >>> from openstef_core.datasets import VersionedTimeSeriesDataset
        >>> from openstef_models.transforms.time_domain import CyclicFeaturesTransform
        >>>
        >>> # Create a versioned time series dataset
        >>> data = pd.DataFrame({
        ...     "timestamp": pd.to_datetime([
        ...         "2025-01-01T10:00:00", "2025-01-01T11:00:00", "2025-01-01T12:00:00"
        ...     ]),
        ...     "available_at": pd.to_datetime([
        ...         "2025-01-01T10:05:00", "2025-01-01T11:05:00", "2025-01-01T12:05:00"
        ...     ]),
        ...     "load": [100.0, 110.0, 120.0],
        ...     "temperature": [20.0, 21.0, 22.0],
        ... })
        >>> versioned_dataset = VersionedTimeSeriesDataset.from_dataframe(data, timedelta(hours=1))
        >>>
        >>> # Configure pipeline for multiple forecast horizons
        >>> pipeline = FeatureEngineeringPipeline(
        ...     horizons=[LeadTime.from_string("PT1H"), LeadTime.from_string("PT24H")],
        ...     versioned_transforms=[],  # No versioned transforms available yet
        ...     horizon_transforms=[
        ...         CyclicFeaturesTransform(included_features=["timeOfDay", "season"])
        ...     ]
        ... )
        >>>
        >>> # Process the versioned dataset through the pipeline
        >>> horizon_datasets = pipeline.fit_transform(versioned_dataset)
        >>> len(horizon_datasets)
        2
        >>> list(horizon_datasets.keys())
        [LeadTime(datetime.timedelta(seconds=3600)), LeadTime(datetime.timedelta(days=1))]
        >>>
        >>> # Check what columns are created (original + cyclic features)
        >>> sorted(horizon_datasets[LeadTime.from_string("PT1H")].feature_names)
        ['load', 'season_cosine', 'season_sine', 'temperature', 'timeOfDay_cosine', 'timeOfDay_sine']

        **Example 2: Pipeline with simple time series dataset (single horizon)**

        Processing a simple dataset for single-horizon forecasting:

        >>> from openstef_core.datasets import TimeSeriesDataset
        >>>
        >>> # Create a simple time series dataset
        >>> simple_data = pd.DataFrame(
        ...     {"load": [100.0, 110.0, 120.0], "temperature": [20.0, 21.0, 22.0]},
        ...     index=pd.date_range("2025-01-01 10:00", periods=3, freq="1h")
        ... )
        >>> simple_dataset = TimeSeriesDataset(simple_data, timedelta(hours=1))
        >>>
        >>> # Configure pipeline for single horizon (no versioned transforms allowed)
        >>> single_pipeline = FeatureEngineeringPipeline(
        ...     horizons=[LeadTime.from_string("PT36H")],
        ...     horizon_transforms=[
        ...         CyclicFeaturesTransform(included_features=["timeOfDay"])
        ...     ]
        ... )
        >>>
        >>> # Process the simple dataset through the pipeline
        >>> single_result = single_pipeline.fit_transform(simple_dataset)
        >>> len(single_result)
        1
        >>> sorted(single_result[LeadTime.from_string("PT36H")].feature_names)
        ['load', 'temperature', 'timeOfDay_cosine', 'timeOfDay_sine']
    """

    horizons: list[LeadTime] = Field(
        default_factory=lambda: [LeadTime.from_string("PT36H")],
        description="The lead times (horizons) for which the model will make predictions.",
        min_length=1,
    )

    versioned_transforms: list[SelfTransform[VersionedTimeSeriesDataset]] = Field(
        default=[],
        description=(
            "Transforms that operate on versioned time series, and usually involve complex time handling logic."
        ),
    )
    horizon_transforms: list[TimeSeriesTransform] = Field(
        default=[], description="Transforms that operate on time series with already resolved timestamps."
    )

    _horizon_split_transform: HorizonSplitTransform = PrivateAttr()
    _is_fitted: bool = PrivateAttr(default=False)

    @override
    def model_post_init(self, context: Any) -> None:
        self._horizon_split_transform = HorizonSplitTransform(horizons=self.horizons)

    def _validate_unversioned_compatibility(self) -> None:
        if len(self.versioned_transforms) > 0:
            raise ValueError("When using unversioned data, the pipeline cannot contain versioned transforms.")
        if len(self.horizons) != 1:
            raise ValueError("When using unversioned data, exactly one horizon must be configured in the pipeline.")

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: VersionedTimeSeriesDataset | TimeSeriesDataset) -> None:
        if isinstance(data, TimeSeriesDataset):
            self._validate_unversioned_compatibility()
            horizon_data = {self.horizons[0]: data}
        else:
            # Fit all the versioned transforms
            versioned_data = data
            for transform in self.versioned_transforms:
                versioned_data = transform.fit_transform(versioned_data)

            # Split to simple time series for fitting simple transforms
            horizon_data = self._horizon_split_transform.fit_transform(versioned_data)

        # Fit all the simple transforms on each simple dataset
        for transform in self.horizon_transforms:
            horizon_data = transform.fit_transform_horizons(horizon_data)

        self._is_fitted = True

    @override
    def transform(self, data: VersionedTimeSeriesDataset | TimeSeriesDataset) -> MultiHorizonTimeSeriesDataset:
        if not self._is_fitted:
            raise TransformNotFittedError("Pipeline is not fitted yet.")

        if isinstance(data, TimeSeriesDataset):
            self._validate_unversioned_compatibility()
            horizon_data = {self.horizons[0]: data}
        else:
            # Apply all the versioned transforms
            versioned_data = data
            for transform in self.versioned_transforms:
                versioned_data = transform.transform(versioned_data)

            # Split to horizon time series for transforming horizon transforms
            horizon_data = self._horizon_split_transform.transform(versioned_data)

        # Transform all the horizon datasets
        for transform in self.horizon_transforms:
            horizon_data = transform.transform_horizons(horizon_data)

        return horizon_data
