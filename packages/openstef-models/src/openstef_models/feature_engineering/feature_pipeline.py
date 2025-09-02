# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Feature engineering pipeline for time series forecasting models.

This module provides the FeaturePipeline class, which coordinates feature engineering
for multi-horizon forecasting models. It handles the complete feature transformation
process, from versioned time series data through horizon-specific transformations.

The pipeline operates in two main phases:
1. Versioned transforms: Handle complex time logic on data with forecast validity timestamps
2. Horizon transforms: Apply standard transformations on resolved time series data

This design enables efficient processing of forecasting data where features may have
different availability times and forecast horizons require different processing.
"""

from typing import Any, override

from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.transforms import TimeSeriesTransform, VersionedTimeSeriesTransform
from openstef_core.exceptions import TransformNotFittedError
from openstef_core.types import LeadTime
from openstef_models.feature_engineering.horizon_split_transform import HorizonSplitTransform


class FeaturePipeline(BaseModel):
    """Feature engineering pipeline for multi-horizon forecasting models.

    Orchestrates feature transformations in a two-stage process optimized for forecasting workflows:

    1. **Versioned transforms**: Applied to raw forecasting data that includes forecast validity timestamps,
       handling complex time logic like feature availability windows and forecast horizon dependencies.

    2. **Horizon transforms**: Applied to resolved time series data for each specific forecast horizon,
       performing standard feature engineering like scaling, aggregation, and encoding.

    This design separates time-aware transformations from standard feature engineering, ensuring
    efficient processing while maintaining forecast data integrity. The pipeline supports both
    versioned and unversioned (single horizon) datasets.

    Example:
        Creating a forecasting pipeline with horizon-specific transforms:

        >>> from openstef_core.types import LeadTime
        >>> from openstef_models.feature_engineering.temporal_transforms import CyclicFeaturesTransform
        >>>
        >>> # Configure pipeline for multiple forecast horizons
        >>> pipeline = FeaturePipeline(
        ...     horizons=[LeadTime.from_string("PT1H"), LeadTime.from_string("PT24H")],
        ...     versioned_transforms=[],  # No versioned transforms available yet
        ...     horizon_transforms=[
        ...         CyclicFeaturesTransform(included_features=["timeOfDay", "season"])
        ...     ]
        ... )
        >>> len(pipeline.horizons)
        2
    """

    horizons: list[LeadTime] = Field(
        default_factory=lambda: [LeadTime.from_string("PT36H")],
        description="The lead times (horizons) for which the model will make predictions.",
    )

    versioned_transforms: list[VersionedTimeSeriesTransform] = Field(
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

    def fit_transform(self, dataset: VersionedTimeSeriesDataset) -> dict[LeadTime, TimeSeriesDataset]:
        """Fit all transforms and apply them to the input dataset.

        Convenience method that combines fitting and transformation in a single call.

        Args:
            dataset: Versioned time series dataset for fitting and transforming.

        Returns:
            Dictionary mapping each lead time to its transformed TimeSeriesDataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def fit(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset) -> None:
        """Fit all pipeline transforms to the input dataset.

        Trains all transforms using the provided dataset. Supports both versioned datasets
        (with complex time handling) and unversioned datasets (for single-horizon use cases).

        Args:
            dataset: Dataset used for fitting transforms. For unversioned data,
                pipeline must have no versioned transforms and exactly one horizon.
        """
        if isinstance(dataset, TimeSeriesDataset):
            self._validate_unversioned_compatibility()
            horizon_datasets = {self.horizons[0]: dataset}
        else:
            # Fit all the versioned transforms
            version_transformed_dataset: VersionedTimeSeriesDataset = dataset
            for transform in self.versioned_transforms:
                version_transformed_dataset = transform.fit_transform(version_transformed_dataset)

            # Split to simple time series for fitting simple transforms
            horizon_datasets: dict[LeadTime, TimeSeriesDataset] = self._horizon_split_transform.transform(
                version_transformed_dataset
            )

        # Fit all the simple transforms on each simple dataset
        for transform in self.horizon_transforms:
            horizon_datasets = transform.fit_transform_horizons(horizon_datasets)

        self._is_fitted = True

    def transform(self, dataset: VersionedTimeSeriesDataset | TimeSeriesDataset) -> dict[LeadTime, TimeSeriesDataset]:
        """Apply fitted transforms to produce horizon-specific datasets.

        Transforms input data through the complete pipeline. Assumes transforms have been fitted.

        Args:
            dataset: Dataset to transform. Can be new data for prediction or training data.

        Returns:
            Dictionary mapping each LeadTime to its transformed TimeSeriesDataset.

        Raises:
            TransformNotFittedError: If pipeline hasn't been fitted before calling this method.
        """
        if not self._is_fitted:
            raise TransformNotFittedError("Pipeline is not fitted yet.")

        if isinstance(dataset, TimeSeriesDataset):
            self._validate_unversioned_compatibility()
            horizon_datasets = {self.horizons[0]: dataset}
        else:
            # Apply all the versioned transforms
            version_transformed_dataset: VersionedTimeSeriesDataset = dataset
            for transform in self.versioned_transforms:
                version_transformed_dataset = transform.transform(version_transformed_dataset)

            # Split to horizon time series for transforming horizon transforms
            horizon_datasets: dict[LeadTime, TimeSeriesDataset] = self._horizon_split_transform.transform(
                version_transformed_dataset
            )

        # Transform all the horizon datasets
        transformed_datasets: dict[LeadTime, TimeSeriesDataset] = horizon_datasets
        for transform in self.horizon_transforms:
            transformed_datasets = transform.transform_horizons(transformed_datasets)

        return transformed_datasets
