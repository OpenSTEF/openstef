# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Inference backends for foundation-model forecasters.

An :class:`InferenceBackend` isolates *how* a checkpoint is executed
(ONNX Runtime, Torch, ...) behind a single named-tensor contract. Forecasters
*compose* a backend rather than inheriting execution behaviour, so swapping
ONNX for Torch is a configuration choice.

Only the **dependency-free** surface is re-exported here: the
:class:`InferenceBackend` protocol and the execution-provider configs (pure
pydantic). The concrete backends live in their own submodules and import their
heavy dependency at module top level, so importing them is an explicit opt-in:

* ``from openstef_foundation_models.inference.onnx_backend import OnnxBackend`` requires ONNX Runtime.
* ``from openstef_foundation_models.inference.torch_backend import TorchBackend`` requires PyTorch.
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
