# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from abc import abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import cast, override

import pandas as pd
from pydantic import Field, TypeAdapter

from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.evaluation.metric_providers import MetricProvider
from openstef_core.base_model import BaseConfig, read_yaml_config
from openstef_core.datasets import VersionedTimeSeriesDataset
from openstef_core.datasets.mixins import VersionedTimeSeriesMixin
from openstef_core.datasets.versioned_timeseries import concat_featurewise


class TargetProviderConfig(BaseConfig):
    """Configuration specifying data locations and path templates for target providers.

    Path templates support {name} placeholder for target-specific file generation.
    All paths are resolved relative to data_dir.
    """


class TargetProvider[T: BenchmarkTarget, F](BaseConfig):
    """Abstract interface for loading benchmark targets and their associated datasets.

    Guarantees consistent access to targets, measurements, and predictor data across
    different benchmark implementations. All returned datasets maintain temporal
    alignment and consistent sampling intervals.
    """

    @abstractmethod
    def get_targets(self, filter_args: F | None = None) -> list[T]:
        """Load all available benchmark targets.

        Args:
            filter_args: Provider-specific filtering criteria.

        Returns:
            Complete list of targets with validated time constraints and metadata.

        Raises:
            FileNotFoundError: When target data source is inaccessible.
            ValidationError: When target definitions violate constraints.
        """

    @abstractmethod
    def get_measurements_for_target(self, target: T) -> VersionedTimeSeriesDataset:
        """Load ground truth measurements for benchmark evaluation.

        Args:
            target: Must have valid time ranges and exist in data source.

        Returns:
            Dataset with measurements covering at least the benchmark period.

        Raises:
            FileNotFoundError: When measurement data is missing for target.
        """

    @abstractmethod
    def get_predictors_for_target(self, target: T) -> VersionedTimeSeriesMixin:
        """Load predictor features for model training and inference.

        Args:
            target: Must have valid time ranges and location coordinates.

        Returns:
            Dataset with features covering training and benchmark periods.

        Raises:
            FileNotFoundError: When predictor data is incomplete for target.
        """

    @abstractmethod
    def get_metrics_for_target(self, target: T) -> list[MetricProvider]:
        """Returns the list of metrics to use for evaluation of a target.

        Args:
            target: The target to get metrics for

        Returns:
            A list of metric providers to use for evaluating predictions for this target
        """

    @abstractmethod
    def get_evaluation_mask_for_target(self, target: T) -> pd.DatetimeIndex | None:
        """Get the evaluation mask for a target.

        Args:
            target: The target to get the evaluation mask for

        Returns:
            A DatetimeIndex representing the evaluation mask, or None if no mask is defined
        """


class SimpleTargetProvider[T: BenchmarkTarget, F](TargetProvider[T, F]):
    """File-based target provider loading from YAML configs and Parquet datasets.

    Implements standardized file loading with consistent path resolution and
    dataset concatenation. Ensures all datasets maintain identical sampling
    intervals and temporal alignment.
    """

    data_dir: Path = Field(description="Root directory containing all benchmark data files")
    measurements_path_template: str = Field(
        default="load_data_{name}.parquet",
        description="Template for target-specific measurement files with {name} placeholder",
    )
    weather_path_template: str = Field(
        default="weather_data_{name}.parquet",
        description="Template for target-specific weather files with {name} placeholder",
    )
    profiles_path: str = Field(default="profiles_data.parquet", description="Path to shared profiles data file.")
    prices_path: str = Field(default="prices_data.parquet", description="Path to shared prices data file.")
    targets_file: str = Field(default="targets.yaml", description="YAML file containing target definitions")

    use_profiles: bool = Field(
        default=True,
        description="Whether to use shared profiles data for predictors",
    )
    use_prices: bool = Field(
        default=True,
        description="Whether to use shared prices data for predictors",
    )

    data_sample_interval: timedelta = Field(
        default=timedelta(minutes=15),
        description="Temporal resolution for all datasets in this provider, used for alignment",
    )

    metrics: list[MetricProvider] = Field(  # type: ignore[reportUnknownMemberType]
        default_factory=list,
        description="List of metric providers to evaluate target forecasts",
    )

    @property
    def get_target_class(self) -> type[T]:
        """Returns the class type of the target."""
        return cast(type[T], BenchmarkTarget)

    @override
    def get_targets(self, filter_args: F | None = None) -> list[T]:
        targets_path = self.data_dir / self.targets_file
        return read_yaml_config(
            path=targets_path,
            class_type=TypeAdapter(list[self.get_target_class]),
        )

    @override
    def get_metrics_for_target(self, target: T) -> list[MetricProvider]:
        return self.metrics

    def get_measurements_path_for_target(self, target: T) -> Path:
        return self.data_dir / str(target.group_name) / self.measurements_path_template.format(name=target.name)

    def get_measurements_for_target(self, target: T) -> VersionedTimeSeriesDataset:
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.get_measurements_path_for_target(target),
        )

    def get_predictors_for_target(self, target: T) -> VersionedTimeSeriesMixin:
        datasets: list[VersionedTimeSeriesMixin] = [
            self.get_weather_for_target(target),
        ]

        if self.use_profiles:
            datasets.append(self.get_profiles())

        if self.use_prices:
            datasets.append(self.get_prices())

        return concat_featurewise(datasets=datasets, mode="inner")

    def get_weather_path_for_target(self, target: T) -> Path:
        return self.data_dir / str(target.group_name) / self.weather_path_template.format(name=target.name)

    def get_weather_for_target(self, target: T) -> VersionedTimeSeriesDataset:
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.get_weather_path_for_target(target),
        )

    def get_profiles(self) -> VersionedTimeSeriesDataset:
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.data_dir / self.profiles_path,
        )

    def get_prices(self) -> VersionedTimeSeriesDataset:
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.data_dir / self.prices_path,
        )

    @override
    def get_evaluation_mask_for_target(self, target: T) -> pd.DatetimeIndex | None:
        return None
