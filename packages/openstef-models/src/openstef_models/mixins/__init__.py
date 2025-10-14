# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Model-related mixins for machine learning workflows.

This package provides mixin classes for model management, storage, and
callback handling in machine learning pipelines.
"""

from .callbacks import PredictorCallback
from .model_serializer import ModelIdentifier, ModelSerializer

__all__ = ["ModelIdentifier", "ModelSerializer", "PredictorCallback"]
