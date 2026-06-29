# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for metadata-driven execution-provider selection."""

from typing import Literal

import pytest
from pydantic import ValidationError

from openstef_core.types import Quantile
from openstef_foundation_models.inference.provider_selection import (
    DefaultProviderPolicy,
    HostCapabilities,
)
from openstef_foundation_models.inference.providers import (
    CoreMLProvider,
    CpuProvider,
    CudaProvider,
)
from openstef_foundation_models.models.checkpoint import CheckpointMetadata

CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"


def _metadata(
    *,
    precision: Literal["fp32", "fp16", "int8"] = "fp32",
    static_shapes: bool = False,
) -> CheckpointMetadata:
    """Build checkpoint metadata varying only the selection-relevant fields."""
    return CheckpointMetadata(
        model_family="chronos2",
        input_names=["context", "group_ids", "attention_mask"],
        output_name="quantile_preds",
        native_quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        context_length=64,
        output_patch_size=16,
        horizon_patches=2,
        resolution_minutes=15,
        precision=precision,
        static_shapes=static_shapes,
    )


def _host(platform: str, *providers: str) -> HostCapabilities:
    return HostCapabilities(platform=platform, available_providers=frozenset({*providers, CPU}))


def test_host_capabilities_is_frozen() -> None:
    """Host facts are an immutable value — they cannot be mutated after detection."""
    host = _host("linux", CUDA)
    with pytest.raises(ValidationError):
        host.platform = "darwin"


def test_static_fp32_on_macos_with_coreml_picks_coreml_gpu_then_cpu() -> None:
    """The known-good Mac path: a static graph runs on CoreML's GPU, never the ANE, with a CPU fallback."""
    chain = DefaultProviderPolicy().select(_metadata(static_shapes=True), _host("darwin", COREML))
    assert chain == [CoreMLProvider(compute_units="CPUAndGPU"), CpuProvider()]


def test_dynamic_fp32_on_macos_skips_coreml() -> None:
    """A dynamic-shape graph cannot compile on CoreML, so macOS falls back to CPU even when CoreML is present."""
    chain = DefaultProviderPolicy().select(_metadata(static_shapes=False), _host("darwin", COREML))
    assert chain == [CpuProvider()]


def test_static_fp32_on_macos_without_coreml_runtime_uses_cpu() -> None:
    """Without the CoreML runtime available, a static macOS graph still resolves to CPU."""
    chain = DefaultProviderPolicy().select(_metadata(static_shapes=True), _host("darwin"))
    assert chain == [CpuProvider()]


def test_int8_on_macos_skips_coreml_for_cpu() -> None:
    """INT8 (QDQ) cannot be accelerated by CoreML, so a static int8 macOS graph goes straight to CPU."""
    chain = DefaultProviderPolicy().select(_metadata(precision="int8", static_shapes=True), _host("darwin", COREML))
    assert chain == [CpuProvider()]


def test_int8_with_cuda_prefers_cuda() -> None:
    """INT8 is fine on a CUDA GPU when one is present."""
    chain = DefaultProviderPolicy().select(_metadata(precision="int8"), _host("linux", CUDA))
    assert chain == [CudaProvider(), CpuProvider()]


def test_fp32_on_linux_with_cuda_prefers_cuda_then_cpu() -> None:
    """On NVIDIA hardware the default chain is CUDA then CPU; TensorRT stays opt-in and is never selected."""
    chain = DefaultProviderPolicy().select(_metadata(static_shapes=True), _host("linux", CUDA))
    assert chain == [CudaProvider(), CpuProvider()]


def test_fp32_on_linux_without_accelerator_uses_cpu() -> None:
    """With no accelerator available the chain is CPU only."""
    chain = DefaultProviderPolicy().select(_metadata(), _host("linux"))
    assert chain == [CpuProvider()]
