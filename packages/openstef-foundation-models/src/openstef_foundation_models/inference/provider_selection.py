# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Metadata-driven execution-provider selection.

Which ONNX Runtime execution provider is fastest — and even which is *usable* —
depends on both the host (Apple CoreML vs NVIDIA CUDA/TensorRT vs CPU) and on
properties of the checkpoint itself (precision, whether its graph has static
shapes). This module keeps that knowledge in **one replaceable component** rather
than scattered platform ``if``-ladders:

* :class:`HostCapabilities` carries the host facts as injectable data, with a
  single impure :meth:`HostCapabilities.detect` classmethod.
* :class:`ProviderPolicy` is the port; :class:`DefaultProviderPolicy` is the
  adapter that maps ``(checkpoint, host)`` to an ordered provider chain. Users
  with exotic hardware implement their own policy.

Importing this module requires ONNX Runtime (the ``[cpu]`` or ``[gpu]`` extra)
and raises :class:`MissingExtraError` if it is missing.
"""

import platform
from typing import Literal, Protocol, Self

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.exceptions import MissingExtraError
from openstef_foundation_models.inference.providers import (
    CoreMLProvider,
    CpuProvider,
    CudaProvider,
    ExecutionProvider,
)
from openstef_foundation_models.models.checkpoint import CheckpointMetadata

try:
    import onnxruntime as ort
except ImportError as e:
    raise MissingExtraError("onnxruntime", "openstef-foundation-models", install_extra="cpu") from e


class HostCapabilities(BaseConfig):
    """Execution-relevant facts about the host, captured as injectable data.

    Passing host facts into a policy (rather than having the policy call
    ``platform.system()`` itself) keeps selection a pure function of its inputs,
    so it can be unit-tested by constructing a fake host.
    """

    model_config = BaseConfig.model_config | {"frozen": True}

    platform: str = Field(
        description="OS identifier, lower-cased (e.g. 'darwin', 'linux', 'windows').",
    )
    available_providers: frozenset[str] = Field(
        description="Execution provider names ONNX Runtime reports as available on this host.",
    )

    @classmethod
    def detect(cls) -> Self:
        """Detect the host's capabilities from the platform and ONNX Runtime.

        This is the one impure call in the selection path; it is isolated here so
        the policy stays a pure function of injected facts.

        Returns:
            The detected host capabilities.
        """
        return cls(
            platform=platform.system().lower(),
            available_providers=frozenset(ort.get_available_providers()),
        )


class ProviderPolicy(Protocol):
    """Port mapping a checkpoint and host to an ordered execution-provider chain.

    Implement this to encode selection rules for hardware the default policy does
    not cover; pass the implementation to the backend or
    :class:`~openstef_foundation_models.presets.forecasting_workflow.OnnxBackendConfig`.

    A policy-selected chain is enforced *gracefully*: ONNX Runtime silently drops
    accelerators it cannot initialize and falls back to CPU, and a policy chain
    such as ``[CoreML, CPU]`` realizing CoreML is the intended outcome, so a
    warning is logged only if it falls all the way to CPU. A chain the caller
    passes explicitly is enforced *strictly* instead: any requested accelerator
    that is not realized raises.
    """

    def select(self, metadata: CheckpointMetadata, host: HostCapabilities) -> list[ExecutionProvider]:
        """Return the ordered provider chain to try for *metadata* on *host*."""
        ...


class DefaultProviderPolicy(BaseConfig):
    """Default policy mapping ``(checkpoint precision/shape, host)`` to a provider chain.

    Each rule encodes a *measured* hardware conclusion; see the design doc
    ``design-docs/0001`` and the provider benchmark for the rationale. The chain
    is ordered preferred-first with CPU as the final fallback.
    """

    kind: Literal["default"] = Field(default="default", description="Discriminator tag for the policy type.")

    def select(  # noqa: PLR6301  # instance method to satisfy the ProviderPolicy protocol, though stateless here
        self, metadata: CheckpointMetadata, host: HostCapabilities
    ) -> list[ExecutionProvider]:
        """Select an ordered provider chain for *metadata* on *host*.

        Args:
            metadata: The checkpoint's metadata (precision, static-shape-ness).
            host: The detected host capabilities.

        Returns:
            An ordered execution-provider chain, preferred-first, CPU last.
        """
        cuda_ok = "CUDAExecutionProvider" in host.available_providers
        coreml_ok = "CoreMLExecutionProvider" in host.available_providers and metadata.static_shapes

        # int8 (QDQ) runs fast on CPU; CoreML cannot accelerate the quantized ops,
        # so it is skipped entirely. CUDA int8 is fine when a GPU is present.
        if metadata.precision == "int8":
            return [CudaProvider(), CpuProvider()] if cuda_ok else [CpuProvider()]
        # macOS: a static-shape fp16/fp32 graph runs on CoreML, but only on the GPU
        # (MLComputeUnits=ALL/ANE triggers a multi-minute Neural-Engine compile for
        # no inference win — measured).
        if host.platform == "darwin" and coreml_ok:
            return [CoreMLProvider(compute_units="CPUAndGPU"), CpuProvider()]
        # NVIDIA: CUDA with a CPU fallback. TensorRT stays opt-in (engine-build cost
        # and fp16 caveats), so the default never selects it.
        if cuda_ok:
            return [CudaProvider(), CpuProvider()]
        return [CpuProvider()]


__all__ = [
    "DefaultProviderPolicy",
    "HostCapabilities",
    "ProviderPolicy",
]
