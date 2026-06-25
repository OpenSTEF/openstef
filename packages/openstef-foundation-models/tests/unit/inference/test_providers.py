# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the typed execution-provider configs."""

from pathlib import Path

from openstef_foundation_models.inference.providers import (
    CoreMLProvider,
    CpuProvider,
    CudaProvider,
    TensorRTProvider,
)


def test_cpu_provider_to_ort() -> None:
    """CpuProvider compiles to the bare CPU provider with no options."""
    assert CpuProvider().to_ort() == ("CPUExecutionProvider", {})


def test_cuda_provider_to_ort_carries_device_id() -> None:
    """CudaProvider forwards its device id."""
    assert CudaProvider(device_id=2).to_ort() == ("CUDAExecutionProvider", {"device_id": 2})


def test_tensorrt_provider_to_ort_enables_engine_cache_when_set(tmp_path: Path) -> None:
    """TensorRTProvider turns on engine caching only when a cache dir is given."""
    name, options = TensorRTProvider(fp16=True, engine_cache_dir=tmp_path).to_ort()
    assert name == "TensorrtExecutionProvider"
    assert options["trt_fp16_enable"] is True
    assert options["trt_engine_cache_enable"] is True
    assert options["trt_engine_cache_path"] == str(tmp_path)


def test_coreml_provider_to_ort_defaults() -> None:
    """CoreMLProvider defaults to MLProgram + ALL with no cache directory."""
    name, options = CoreMLProvider().to_ort()
    assert name == "CoreMLExecutionProvider"
    assert options == {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    assert "ModelCacheDirectory" not in options


def test_coreml_provider_to_ort_includes_cache_dir_when_set(tmp_path: Path) -> None:
    """A configured cache_dir is emitted as the ORT ModelCacheDirectory option."""
    name, options = CoreMLProvider(compute_units="CPUAndGPU", cache_dir=tmp_path).to_ort()
    assert name == "CoreMLExecutionProvider"
    assert options["MLComputeUnits"] == "CPUAndGPU"
    assert options["ModelCacheDirectory"] == str(tmp_path)
