# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

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
        benchmark_start = info.data.get("benchmark_start")
        if benchmark_start and v <= benchmark_start:
            raise ValueError("benchmark_end must be after benchmark_start")
        return v

    @field_validator("train_start")
    @classmethod
    def validate_train_start(cls, v: datetime, info: ValidationInfo) -> datetime:
        benchmark_start = info.data.get("benchmark_start")
        if benchmark_start and v >= benchmark_start:
            raise ValueError("train_start must be before benchmark_start")
        return v
