# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Benchmark target models for defining forecasting evaluation scenarios.

Provides base classes and data structures for representing benchmark targets,
which define the scope, timing, and characteristics of forecasting evaluation
tasks. Each target represents a specific forecasting challenge with defined
training and evaluation periods.
"""

from datetime import datetime

from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from openstef_core.base_model import BaseConfig


class BenchmarkTarget(BaseConfig):
    """Base class for benchmark targets with common properties.

    Defines the core properties that all benchmark targets must have, ensuring
    a consistent interface across different benchmark implementations.

    Raises:
        ValueError: When benchmark_end <= benchmark_start or train_start >= benchmark_start.
    """

    name: str = Field(description="Unique identifier for the benchmark target")
    description: str = Field(description="Human-readable description of the target")
    group_name: str = Field(default="default", description="Group name for categorizing targets")

    latitude: float = Field(description="Geographical latitude of the target location")
    longitude: float = Field(description="Geographical longitude of the target location")
    limit: float = Field(description="Capacity limit of the target in appropriate units")

    benchmark_start: datetime = Field(description="Start timestamp of the benchmark period")
    benchmark_end: datetime = Field(description="End timestamp of the benchmark period")
    train_start: datetime = Field(description="Start timestamp of the training period")

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
