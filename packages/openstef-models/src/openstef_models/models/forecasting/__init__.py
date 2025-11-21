# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecasting interfaces and implementations for OpenSTEF models.

This module provides the core forecasting abstractions and concrete implementations.
The base interfaces define the contract for all forecasters, while specific
implementations demonstrate different forecasting approaches.

Interfaces:
    - BaseForecaster: Core multi-horizon forecasting interface
    - BaseHorizonForecaster: Single-horizon forecasting interface
    - Configuration classes for forecaster setup and validation

Implementations:
    - constant_median_forecaster: Simple baseline forecaster using historical medians
    - multi_horizon_adapter: Adapter pattern for converting single to multi-horizon forecasters
"""

from .forecaster import Forecaster, ForecasterConfig

__all__ = [
    "Forecaster",
    "ForecasterConfig",
]
