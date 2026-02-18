# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Core mixins for building reusable components.

This package provides fundamental mixin classes that enable common patterns
across the OpenSTEF codebase, including prediction interfaces, state management,
and data transformation pipelines.
"""

from .forecaster import Forecaster, ForecasterConfig
from .predictor import BatchPredictor, BatchResult, HyperParams, Predictor
from .stateful import Stateful
from .transform import Transform, TransformPipeline

__all__ = [
    "BatchPredictor",
    "BatchResult",
    "Forecaster",
    "ForecasterConfig",
    "HyperParams",
    "Predictor",
    "Stateful",
    "Transform",
    "TransformPipeline",
]
