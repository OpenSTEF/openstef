# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Target provider interfaces and implementations for benchmark execution.

Target providers are responsible for loading benchmark targets and their associated datasets
(measurements, predictors, and evaluation configuration). They ensure data consistency,
temporal alignment, and provide a standardized interface for accessing benchmark data
across different sources and formats.

The module supports both simple file-based providers and complex database-backed
implementations through abstract interfaces that guarantee consistent behavior.
"""

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


class TargetProviderConfig(BaseConfig):
    """Configuration specifying data locations and path templates for target providers.

    Path templates support {name} placeholder for target-specific file generation.
    All paths are resolved relative to data_dir.

    Examples:
        Basic configuration with default templates:

        >>> config = TargetProviderConfig()
        >>> # Uses default values, can be extended by specific providers

        Custom path configuration:

        >>> config = TargetProviderConfig()
        >>> # Provider-specific configs inherit from this base
    """


class TargetProvider[T: BenchmarkTarget, F](BaseConfig):
    """Abstract interface for loading benchmark targets and their associated datasets.

    Guarantees consistent access to targets, measurements, and predictor data across
    different benchmark implementations. All returned datasets maintain temporal
    alignment and consistent sampling intervals.

    Examples:
        Complete custom provider implementation:

        >>> from pathlib import Path
        >>> from datetime import datetime, timedelta
        >>> from openstef_beam.benchmarking.models.benchmark_target import BenchmarkTarget
        >>> from openstef_beam.evaluation.metric_providers import RMAEProvider
        >>> class EnergyTargetProvider(TargetProvider[BenchmarkTarget, None]):
        ...     def __init__(self, data_path: Path, region: str):
        ...         super().__init__()
        ...         self.data_path = data_path
        ...         self.region = region
        ...
        ...     def get_targets(self, filter_args=None):
        ...         # Load from database or config files
        ...         return [
        ...             BenchmarkTarget(
        ...                 name=f"substation_{i:03d}",
        ...                 description=f"Energy load for substation {i}",
        ...                 group_name=self.region,
        ...                 latitude=52.0 + i * 0.001,
        ...                 longitude=4.0 + i * 0.001,
        ...                 limit=100.0 + i * 10,
        ...                 benchmark_start=datetime(2024, 1, 1),
        ...                 benchmark_end=datetime(2024, 3, 1),
        ...                 train_start=datetime(2022, 1, 1)
        ...             ) for i in range(1, 11)
        ...         ]
        ...
        ...     def get_measurements_for_target(self, target):
        ...         # Load actual load data from parquet files
        ...         return VersionedTimeSeriesDataset.read_parquet(
        ...             self.data_path / f"{target.group_name}/{target.name}_load.parquet"
        ...         )
        ...
        ...     def get_predictors_for_target(self, target):
        ...         # Combine weather, profiles, and building characteristics
        ...         datasets = [
        ...             self.load_weather_data(target),
        ...             self.load_building_profiles(target),
        ...             self.load_price_data()
        ...         ]
        ...         return concat_featurewise(datasets, mode="inner")
        ...
        ...     def get_metrics_for_target(self, target):
        ...         # Target-specific metrics based on building type
        ...         base_metrics = [RMAEProvider(), RCRPSProvider()]
        ...         if target.limit > 500:  # Large buildings get additional metrics
        ...             base_metrics.append(MAPEProvider())
        ...         return base_metrics
    """

    target_column: str = Field(
        default="load",
        description="Name of the target column in the ground truth dataset",
    )

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
    def get_predictors_for_target(self, target: T) -> VersionedTimeSeriesDataset:
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

    Directory structure expected by SimpleTargetProvider:

    The provider expects a hierarchical directory structure:
    - Root directory contains shared data files and target definitions
    - Group subdirectories contain target-specific measurement and weather files
    - Path templates use {name} placeholder for target-specific file naming

    Examples:
        Complete provider setup with data loading:

        >>> from pathlib import Path
        >>> from datetime import timedelta
        >>> from openstef_beam.evaluation.metric_providers import RMAEProvider, RCRPSProvider
        >>> provider = SimpleTargetProvider(
        ...     data_dir=Path("./benchmark_data"),
        ...     measurements_path_template="demand_{name}.parquet",
        ...     weather_path_template="weather_{name}.parquet",
        ...     profiles_path="standard_profiles.parquet",
        ...     prices_path="energy_prices.parquet",
        ...     targets_file="energy_targets.yaml",
        ...     data_sample_interval=timedelta(minutes=15),
        ...     metrics=[RMAEProvider(), RCRPSProvider()],
        ...     use_profiles=True,
        ...     use_prices=True
        ... )
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
        """Build file path for target measurements using configured template.

        Returns:
            Path: Path to the measurements file for the target.
        """
        return self.data_dir / str(target.group_name) / self.measurements_path_template.format(name=target.name)

    def get_measurements_for_target(self, target: T) -> VersionedTimeSeriesDataset:
        """Load ground truth measurements from target-specific Parquet file.

        Returns:
            VersionedTimeSeriesDataset: The loaded measurements data.
        """
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.get_measurements_path_for_target(target),
        )

    def get_predictors_for_target(self, target: T) -> VersionedTimeSeriesDataset:
        """Combine weather, profiles, and prices into aligned predictor dataset.

        Concatenates datasets feature-wise with inner join to ensure temporal alignment.
        Only includes datasets that are enabled via configuration flags.

        Returns:
            VersionedTimeSeriesMixin: Combined predictor dataset with all enabled features.
        """
        datasets: list[VersionedTimeSeriesDataset] = [
            self.get_weather_for_target(target),
        ]

        if self.use_profiles:
            datasets.append(self.get_profiles())

        if self.use_prices:
            datasets.append(self.get_prices())

        return VersionedTimeSeriesDataset.concat(datasets, mode="inner")

    def get_weather_path_for_target(self, target: T) -> Path:
        """Build file path for target weather data using configured template.

        Returns:
            Path: Path to the weather data file for the target.
        """
        return self.data_dir / str(target.group_name) / self.weather_path_template.format(name=target.name)

    def get_weather_for_target(self, target: T) -> VersionedTimeSeriesDataset:
        """Load weather features from target-specific Parquet file.

        Returns:
            VersionedTimeSeriesDataset: The loaded weather data.
        """
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.get_weather_path_for_target(target),
        )

    def get_profiles(self) -> VersionedTimeSeriesDataset:
        """Load shared energy profiles data from configured Parquet file.

        Returns:
            VersionedTimeSeriesDataset: The loaded energy profiles data.
        """
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.data_dir / self.profiles_path,
        )

    def get_prices(self) -> VersionedTimeSeriesDataset:
        """Load shared energy pricing data from configured Parquet file.

        Returns:
            VersionedTimeSeriesDataset: The loaded energy pricing data.
        """
        return VersionedTimeSeriesDataset.read_parquet(
            path=self.data_dir / self.prices_path,
        )

    @override
    def get_evaluation_mask_for_target(self, target: T) -> pd.DatetimeIndex | None:
        return None
