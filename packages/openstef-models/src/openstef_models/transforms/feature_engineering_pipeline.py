# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Feature engineering pipeline for time series forecasting models.

This module provides the FeaturePipeline class, which coordinates feature engineering
for multi-horizon forecasting models. It handles the complete feature transformation
process, from versioned time series data through horizon-specific transformations.
"""

from collections.abc import Sequence
from typing import Any, Self, cast, override

from pydantic import BaseModel, Field, PrivateAttr

from openstef_core.datasets import (
    MultiHorizon,
    TimeSeriesDataset,
    VersionedTimeSeriesDataset,
)
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import State, Transform, TransformPipeline
from openstef_core.transforms import MultiHorizonTimeSeriesTransform, TimeSeriesTransform
from openstef_core.types import LeadTime
from openstef_models.transforms import HorizonSplitTransform


class FeatureEngineeringPipeline(
    BaseModel, Transform[VersionedTimeSeriesDataset | TimeSeriesDataset, MultiHorizon[TimeSeriesDataset]]
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
        >>> from openstef_models.transforms.time_domain import CyclicFeaturesAdder
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
        >>> pipeline = FeatureEngineeringPipeline.create(
        ...     horizons=[LeadTime.from_string("PT1H"), LeadTime.from_string("PT24H")],
        ...     versioned_transforms=[],  # No versioned transforms available yet
        ...     horizon_transforms=[
        ...         CyclicFeaturesAdder(included_features=["time_of_day", "season"])
        ...     ]
        ... )
        >>>
        >>> # Process the versioned dataset through the pipeline
        >>> horizon_datasets = pipeline.fit_transform(versioned_dataset)
        >>> len(horizon_datasets)
        2
        >>> list(horizon_datasets.keys())
        [LeadTime('PT1H'), LeadTime('P1D')]
        >>>
        >>> # Check what columns are created (original + cyclic features)
        >>> sorted(horizon_datasets[LeadTime.from_string("PT1H")].feature_names)
        ['load', 'season_cosine', 'season_sine', 'temperature', 'time_of_day_cosine', 'time_of_day_sine']

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
        >>> single_pipeline = FeatureEngineeringPipeline.create(
        ...     horizons=[LeadTime.from_string("PT36H")],
        ...     horizon_transforms=[
        ...         CyclicFeaturesAdder(included_features=["time_of_day"])
        ...     ]
        ... )
        >>>
        >>> # Process the simple dataset through the pipeline
        >>> single_result = single_pipeline.fit_transform(simple_dataset)
        >>> len(single_result)
        1
        >>> sorted(single_result[LeadTime.from_string("PT36H")].feature_names)
        ['load', 'temperature', 'time_of_day_cosine', 'time_of_day_sine']
    """

    horizons: list[LeadTime] = Field(
        default_factory=lambda: [LeadTime.from_string("PT36H")],
        description="The lead times (horizons) for which the model will make predictions.",
        min_length=1,
    )

    versioned_pipeline: TransformPipeline[VersionedTimeSeriesDataset] = Field(
        default_factory=lambda: TransformPipeline(transforms=[]),
        description=(
            "Transforms that operate on versioned time series, and usually involve complex time handling logic."
        ),
    )
    horizon_pipeline: TransformPipeline[MultiHorizon[TimeSeriesDataset]] = Field(
        default_factory=lambda: TransformPipeline(transforms=[]),
        description="Transforms that operate on time series with already resolved timestamps.",
    )

    _horizon_split_transform: HorizonSplitTransform = PrivateAttr()

    @classmethod
    def create(
        cls,
        versioned_transforms: Sequence[Transform[VersionedTimeSeriesDataset, VersionedTimeSeriesDataset]] | None = None,
        horizon_transforms: Sequence[TimeSeriesTransform | MultiHorizonTimeSeriesTransform] | None = None,
        horizons: list[LeadTime] | None = None,
    ) -> Self:
        """Create a feature engineering pipeline with specified transforms.

        Factory method to construct a pipeline with versioned and horizon-specific
        transforms. Automatically adapts single-horizon transforms to multi-horizon
        format when needed.

        Args:
            versioned_transforms: Transforms that operate on versioned time series data.
            horizon_transforms: Transforms that operate on horizon-specific data.
                Single-horizon transforms are automatically adapted to multi-horizon.
            horizons: Forecast horizons for the pipeline. Defaults to 36 hours.

        Returns:
            Configured feature engineering pipeline instance.
        """
        return cls(
            horizons=horizons or [LeadTime.from_string("PT36H")],
            versioned_pipeline=TransformPipeline(transforms=versioned_transforms or []),
            horizon_pipeline=TransformPipeline(
                transforms=[
                    transform
                    if isinstance(transform, MultiHorizonTimeSeriesTransform)
                    else transform.to_multi_horizon()
                    for transform in horizon_transforms or []
                ]
            ),
        )

    @override
    def model_post_init(self, context: Any) -> None:
        self._horizon_split_transform = HorizonSplitTransform(horizons=self.horizons)

    @override
    def to_state(self) -> State:
        return {
            "versioned_transforms": self.versioned_pipeline.to_state(),
            "horizon_transforms": self.horizon_pipeline.to_state(),
        }

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, State], state)

        return self.__class__(
            horizons=self.horizons,
            versioned_pipeline=self.versioned_pipeline.from_state(state["versioned_transforms"]),
            horizon_pipeline=self.horizon_pipeline.from_state(state["horizon_transforms"]),
        )

    @property
    @override
    def is_fitted(self) -> bool:
        return self.versioned_pipeline.is_fitted and self.horizon_pipeline.is_fitted

    def _validate_unversioned_compatibility(self) -> None:
        if len(self.versioned_pipeline.transforms) > 0:
            raise ValueError("When using unversioned data, the pipeline cannot contain versioned transforms.")

    @override
    def fit(self, data: VersionedTimeSeriesDataset | TimeSeriesDataset) -> None:
        if isinstance(data, TimeSeriesDataset):
            horizon_data = MultiHorizon(dict.fromkeys(self.horizons, data))
        else:
            # Fit all the versioned transforms
            versioned_data = self.versioned_pipeline.fit_transform(data=data)

            # Split to simple time series into horizon-specific datasets
            horizon_data = self._horizon_split_transform.fit_transform(data=versioned_data)

        # Fit all the horizon transforms
        self.horizon_pipeline.fit_transform(data=horizon_data)

    @override
    def transform(self, data: VersionedTimeSeriesDataset | TimeSeriesDataset) -> MultiHorizon[TimeSeriesDataset]:
        if not self.is_fitted:
            raise NotFittedError("Pipeline is not fitted")

        if isinstance(data, TimeSeriesDataset):
            horizon_data = MultiHorizon(dict.fromkeys(self.horizons, data))
        else:
            versioned_data = self.versioned_pipeline.transform(data=data)
            horizon_data = self._horizon_split_transform.transform(data=versioned_data)

        return self.horizon_pipeline.transform(data=horizon_data)
