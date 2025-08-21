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
from typing import Self

import yaml
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, TypeAdapter


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


__all__ = [
    "BaseConfig",
    "BaseModel",
    "read_yaml_config",
    "write_yaml_config",
]
