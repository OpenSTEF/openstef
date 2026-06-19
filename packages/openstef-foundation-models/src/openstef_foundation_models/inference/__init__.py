# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Inference backends for foundation-model forecasters.

An :class:`InferenceBackend` isolates how a checkpoint is executed behind a
single named-tensor contract; forecasters compose a backend rather than
inheriting execution behaviour. Only the dependency-free surface (the protocol
and the execution-provider configs) is re-exported here; a concrete backend
lives in its own submodule and imports its heavy runtime at module top level.
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
