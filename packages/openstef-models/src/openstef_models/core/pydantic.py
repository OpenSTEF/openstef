# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core Pydantic models and configuration for OpenSTEF.

This module provides base classes for Pydantic models used throughout the OpenSTEF project,
ensuring consistent configuration and serialization behavior.
"""

from pydantic import BaseModel as BaseModelPydantic
from pydantic import ConfigDict


class BaseModel(BaseModelPydantic):
    """Base model for OpenSTEF components."""

    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True, ser_json_inf_nan="null")


class BaseConfig(BaseModelPydantic):
    """Base configuration model.

    It configures Pydantic model for safe serialization / deserialization.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="ignore", arbitrary_types_allowed=False)


__all__ = [
    "BaseConfig",
    "BaseModel",
]
