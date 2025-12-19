# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecasting workflow presets package.

Provides configurations and utilities for setting up forecasting workflows.
"""

from .forecasting_workflow import ForecastingWorkflowConfig, create_forecasting_workflow

__all__ = [
    "ForecastingWorkflowConfig",
    "create_forecasting_workflow",
]
