# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Benchmark target models for defining forecasting evaluation scenarios.

Provides base classes and data structures for representing benchmark targets,
which define the scope, timing, and characteristics of forecasting evaluation
tasks. Each target represents a specific forecasting challenge with defined
training and evaluation periods.
"""

from datetime import datetime

from pydantic import Field, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo
from pydantic_extra_types.coordinate import Latitude, Longitude

from openstef_core.base_model import BaseConfig


class BenchmarkTarget(BaseConfig):
    """Base class for benchmark targets with common properties.

    Defines the core properties that all benchmark targets must have, ensuring
    a consistent interface across different benchmark implementations.

    Raises:
        ValueError: When benchmark_end <= benchmark_start, train_start >= benchmark_start,
                   or when limit constraints are not met (either 'limit' alone or both
                   'upper_limit' and 'lower_limit' must be specified).
    """

    name: str = Field(description="Unique identifier for the benchmark target")
    description: str = Field(description="Human-readable description of the target")
    group_name: str = Field(default="default", description="Group name for categorizing targets")

    latitude: Latitude = Field(description="Geographical latitude of the target location")
    longitude: Longitude = Field(description="Geographical longitude of the target location")
    limit: float | None = Field(default=None, description="Capacity limit of the target in appropriate units")
    upper_limit: float | None = Field(
        default=None, description="Upper capacity limit of the target in appropriate units"
    )
    lower_limit: float | None = Field(
        default=None, description="Lower capacity limit of the target in appropriate units"
    )

    benchmark_start: datetime = Field(description="Start timestamp of the benchmark period")
    benchmark_end: datetime = Field(description="End timestamp of the benchmark period")
    train_start: datetime = Field(description="Start timestamp of the training period")

    @model_validator(mode="after")
    def validate_limits(self) -> "BenchmarkTarget":
        """Validate that either limit or both upper_limit and lower_limit are provided.

        Returns:
            The validated BenchmarkTarget instance.

        Raises:
            ValueError: If neither limit nor (upper_limit and lower_limit) are provided,
                       or if both limit and (upper_limit or lower_limit) are provided.
        """
        has_limit = self.limit is not None
        has_upper = self.upper_limit is not None
        has_lower = self.lower_limit is not None

        if has_limit and (has_upper or has_lower):
            raise ValueError(
                "Cannot specify both 'limit' and 'upper_limit'/'lower_limit'. "
                "Use either 'limit' alone or both 'upper_limit' and 'lower_limit'."
            )

        if not has_limit and not (has_upper and has_lower):
            raise ValueError("Must specify either 'limit' or both 'upper_limit' and 'lower_limit'.")

        return self

    @field_validator("benchmark_end")
    @classmethod
    def validate_benchmark_end(cls, v: datetime, info: ValidationInfo) -> datetime:
        """Validate that benchmark_end occurs after benchmark_start.

        Args:
            v: The benchmark_end value to validate.
            info: Validation context containing other field values.

        Returns:
            The validated benchmark_end datetime.

        Raises:
            ValueError: If benchmark_end is not after benchmark_start.
        """
        benchmark_start = info.data.get("benchmark_start")
        if benchmark_start and v <= benchmark_start:
            raise ValueError("benchmark_end must be after benchmark_start")
        return v

    @field_validator("train_start")
    @classmethod
    def validate_train_start(cls, v: datetime, info: ValidationInfo) -> datetime:
        """Validate that train_start occurs before benchmark_start.

        Args:
            v: The train_start value to validate.
            info: Validation context containing other field values.

        Returns:
            The validated train_start datetime.

        Raises:
            ValueError: If train_start is not before benchmark_start.
        """
        benchmark_start = info.data.get("benchmark_start")
        if benchmark_start and v >= benchmark_start:
            raise ValueError("train_start must be before benchmark_start")
        return v
