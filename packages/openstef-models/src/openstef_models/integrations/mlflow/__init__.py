# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""MLflow integration for model tracking and storage.

Provides integration with MLflow for model lifecycle management, experiment
tracking, and model registry functionality. This package enables OpenSTEF
models to be stored, versioned, and tracked using MLflow's
model registry.

Note:
    This package requires MLflow to be installed as an optional dependency.
    MLflow integration is particularly useful for production deployments
    requiring model versioning, experiment tracking, and centralized storage.
"""

from .mlflow_storage import MLFlowStorage
from .mlflow_storage_callback import MLFlowStorageCallback

__all__ = ["MLFlowStorage", "MLFlowStorageCallback"]
