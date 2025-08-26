# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Configuration utilities for OpenSTEF Beam.

This module provides a `BaseConfig` class extending Pydantic's `BaseModel`
with convenience helpers for reading from and writing to YAML files. It also
exposes two helper functions `write_yaml_config` and `read_yaml_config` that
operate on arbitrary config instances or Pydantic models / adapters.
"""

from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel as PydanticBaseModel
from pydantic import BeforeValidator, ConfigDict, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import core_schema


class BaseModel(PydanticBaseModel):
    """Base model class for OpenSTEF components."""

    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True, ser_json_inf_nan="null")


class BaseConfig(PydanticBaseModel):
    """Base configuration model.

    It configures Pydantic model for safe YAML serialization / deserialization.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",
        arbitrary_types_allowed=False,
    )

    @classmethod
    def read_yaml(cls, path: Path) -> Self:
        """Create an instance from a YAML file.

        Args:
            path: Path to the YAML file to read.

        Returns:
            An instance of the config class populated with the file contents.
        """
        return read_yaml_config(path, class_type=cls)

    def write_yaml(self, path: Path) -> None:
        """Write this configuration to a YAML file.

        Args:
            path: Destination path for the YAML file (will be overwritten).
        """
        write_yaml_config(self, path)


def write_yaml_config(config: BaseConfig, path: Path) -> None:
    """Write the config to a YAML file.

    Args:
        config: The configuration object to serialize.
        path: Destination path for the YAML file (will be overwritten).

    Example:
        >>> from pathlib import Path
        >>> from pydantic import BaseModel
        >>> class MyConfig(BaseModel):
        ...     foo: int
        >>> cfg = MyConfig(foo=123)
        >>> write_yaml_config(cfg, Path("/tmp/test.yaml"))
    """
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(mode="json"), f, allow_unicode=True)


def read_yaml_config[T: BaseConfig, U](path: Path, class_type: type[T] | TypeAdapter[U]) -> T | U:
    """Read a configuration object from a YAML file.

    This function supports two kinds of targets:

    * A subclass of `BaseConfig`, in which case Pydantic's `model_validate` is used.
    * A `TypeAdapter` instance for more advanced / non-`BaseModel` schema validation.

    Args:
        path: Path to the YAML file to read.
        class_type: The target type (a `BaseConfig` subclass) or a `TypeAdapter`.

    Returns:
        A validated configuration instance (either ``T`` or ``U`` depending on
        the provided ``class_type``).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(class_type, TypeAdapter):
        return class_type.validate_python(data)

    return class_type.model_validate(data)


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
        """Validate and convert input to this type.

        Args:
            v: Input value to validate.
            _info: Additional validation info (unused).

        Returns:
            Validated instance of this type.

        Raises:
            ValueError: If input cannot be converted to this type.
        """
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
        """Define Pydantic validation and serialization behavior.

        Returns:
            Core schema for Pydantic validation and serialization.
        """
        return core_schema.with_info_plain_validator_function(
            function=cls.validate, serialization=core_schema.plain_serializer_function_ser_schema(cls.__str__)
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on string representation.

        Returns:
            True if both objects have the same string representation, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return str(self) == str(other)

    def __hash__(self) -> int:
        """Return hash based on string representation."""
        return hash(str(self))


def _convert_none_to_nan(v: float | None) -> float:
    if v is None:
        return float("nan")
    return v


FloatOrNan = Annotated[float, BeforeValidator(_convert_none_to_nan)]

__all__ = [
    "BaseConfig",
    "BaseModel",
    "FloatOrNan",
    "PydanticStringPrimitive",
    "read_yaml_config",
    "write_yaml_config",
]
