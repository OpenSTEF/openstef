# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from typing import Any, Self

from pydantic import TypeAdapter
from pydantic._internal._generate_schema import GetCoreSchemaHandler
from pydantic_core import core_schema


class PydanticStringPrimitive:
    """Base class for Pydantic-compatible types with string serialization."""

    def __str__(self) -> str:
        """Convert to string representation."""
        raise NotImplementedError("Subclasses must implement __str__")

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Create an instance from string representation."""
        raise NotImplementedError("Subclasses must implement from_string")

    @classmethod
    def validate(cls, v: Any, _info: Any = None) -> Self:  # noqa: ANN401
        """Validate and convert input to this type."""
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            return cls.from_string(v)

        # Subclasses should handle their specific types
        error_message = f"Cannot convert {v} to {cls.__name__}"
        raise ValueError(error_message)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: type[Any], _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Define Pydantic validation and serialization behavior."""
        return core_schema.with_info_plain_validator_function(
            function=cls.validate, serialization=core_schema.plain_serializer_function_ser_schema(cls.__str__)
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on string representation."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return str(self) == str(other)

    def __hash__(self) -> int:
        """Return hash based on string representation."""
        return hash(str(self))


def timedelta_to_isoformat(td: timedelta) -> str:
    """Convert timedelta to ISO 8601 string format."""
    return TypeAdapter(timedelta).dump_python(td, mode="json")


def timedelta_from_isoformat(s: str) -> timedelta:
    """Convert ISO 8601 string format to timedelta."""
    return TypeAdapter(timedelta).validate_python(s)
