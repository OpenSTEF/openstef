# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Presets for building foundation-model forecasters from declarative config.

This package is import-light: the heavy inference runtime is imported lazily
when a backend config is built, not when these presets are imported.
"""

from openstef_foundation_models.presets.forecasting_workflow import (
    BackendConfig,
    ForecastingWorkflowConfig,
    OnnxBackendConfig,
    create_forecasting_workflow,
)

__all__ = [
    "BackendConfig",
    "ForecastingWorkflowConfig",
    "OnnxBackendConfig",
    "create_forecasting_workflow",
]
