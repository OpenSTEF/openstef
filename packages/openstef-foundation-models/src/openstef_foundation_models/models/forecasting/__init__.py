# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Foundation-model forecasters.

These forecasters compose an
:class:`~openstef_foundation_models.inference.backend.InferenceBackend` and are
themselves dependency-free: the heavy runtime (ONNX Runtime) lives in the
injected backend, not here.
"""

from openstef_foundation_models.models.forecasting.chronos2_forecaster import (
    Chronos2Forecaster,
    Chronos2HyperParams,
)

__all__ = [
    "Chronos2Forecaster",
    "Chronos2HyperParams",
]
