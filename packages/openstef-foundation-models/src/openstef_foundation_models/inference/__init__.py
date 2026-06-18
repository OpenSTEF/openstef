# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Inference backends for foundation-model forecasters.

An :class:`InferenceBackend` isolates *how* a checkpoint is executed behind a
single named-tensor contract. Forecasters *compose* a backend rather than
inheriting execution behaviour, so the execution runtime is a configuration
choice. ONNX Runtime is the only backend today; the contract keeps room for
others without touching forecaster code.

Only the **dependency-free** surface is re-exported here: the
:class:`InferenceBackend` protocol and the execution-provider configs (pure
pydantic). The concrete backend lives in its own submodule and imports its
heavy dependency at module top level, so importing it is an explicit opt-in:

* ``from openstef_foundation_models.inference.onnx_backend import OnnxBackend`` requires ONNX Runtime.
"""

from openstef_foundation_models.inference.backend import InferenceBackend
from openstef_foundation_models.inference.providers import (
    CoreMLProvider,
    CpuProvider,
    CudaProvider,
    ExecutionProvider,
    SessionOptionsConfig,
    TensorRTProvider,
)

__all__ = [
    "CoreMLProvider",
    "CpuProvider",
    "CudaProvider",
    "ExecutionProvider",
    "InferenceBackend",
    "SessionOptionsConfig",
    "TensorRTProvider",
]
