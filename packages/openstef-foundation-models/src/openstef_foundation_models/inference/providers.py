# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Typed ONNX Runtime execution-provider configuration.

Each provider is a small pydantic config that compiles to an ONNX Runtime
``(name, options)`` tuple via :meth:`ExecutionProviderConfig.to_ort`. Keeping
providers as typed configs (rather than raw strings) lets users opt into
hardware acceleration — CUDA, TensorRT FP16, CoreML/ANE — without touching
model code, and keeps the options validated and discoverable.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from openstef_core.base_model import BaseConfig

#: An ONNX Runtime provider specification: ``(provider_name, provider_options)``.
type OrtProvider = tuple[str, dict[str, object]]


class CpuProvider(BaseConfig):
    """The default CPU execution provider."""

    kind: Literal["cpu"] = Field(default="cpu", description="Discriminator tag for execution-provider type.")

    def to_ort(self) -> OrtProvider:
        """Compile to an ONNX Runtime provider tuple.

        Returns:
            The ``CPUExecutionProvider`` with no options.
        """
        return ("CPUExecutionProvider", {})


class CudaProvider(BaseConfig):
    """The CUDA (NVIDIA GPU) execution provider."""

    kind: Literal["cuda"] = Field(default="cuda", description="Discriminator tag for execution-provider type.")
    device_id: int = Field(default=0, ge=0, description="CUDA device index to run on.")

    def to_ort(self) -> OrtProvider:
        """Compile to an ONNX Runtime provider tuple.

        Returns:
            The ``CUDAExecutionProvider`` with the configured device id.
        """
        return ("CUDAExecutionProvider", {"device_id": self.device_id})


class TensorRTProvider(BaseConfig):
    """The TensorRT execution provider (NVIDIA, ahead-of-time engine build).

    FP16 with a persistent engine cache is the recommended production path on
    NVIDIA hardware: the first run pays the engine-build cost, subsequent runs
    load the cached engine.
    """

    kind: Literal["tensorrt"] = Field(default="tensorrt", description="Discriminator tag for execution-provider type.")
    device_id: int = Field(default=0, ge=0, description="CUDA device index to run on.")
    fp16: bool = Field(default=True, description="Enable FP16 precision for faster inference.")
    engine_cache_dir: Path | None = Field(
        default=None,
        description="Directory to persist built TensorRT engines. When set, engine caching is enabled.",
    )

    def to_ort(self) -> OrtProvider:
        """Compile to an ONNX Runtime provider tuple.

        Returns:
            The ``TensorrtExecutionProvider`` with precision and engine-cache options.
        """
        options: dict[str, object] = {
            "device_id": self.device_id,
            "trt_fp16_enable": self.fp16,
        }
        if self.engine_cache_dir is not None:
            options["trt_engine_cache_enable"] = True
            options["trt_engine_cache_path"] = str(self.engine_cache_dir)
        return ("TensorrtExecutionProvider", options)


class CoreMLProvider(BaseConfig):
    """The CoreML execution provider (Apple, GPU/Neural Engine).

    ``ModelFormat=MLProgram`` is required for modern CoreML: the legacy
    ``NeuralNetwork`` format fragments the graph and silently falls back to CPU
    for many ops.
    """

    kind: Literal["coreml"] = Field(default="coreml", description="Discriminator tag for execution-provider type.")
    model_format: Literal["MLProgram", "NeuralNetwork"] = Field(
        default="MLProgram",
        description="CoreML model format. MLProgram is required for modern op coverage.",
    )
    compute_units: Literal["CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine", "ALL"] = Field(
        default="ALL",
        description="Which compute units CoreML may dispatch to. Prefer 'CPUAndGPU' for large transformer "
        "graphs: allowing the Neural Engine ('ALL'/'CPUAndNeuralEngine') can make CoreML's ahead-of-time "
        "compile run for many minutes for no inference win.",
    )
    cache_dir: Path | None = Field(
        default=None,
        description="Directory for CoreML's compiled-model cache (ORT 'ModelCacheDirectory'). CoreML compiles "
        "the graph ahead of time on session build, which is slow; caching it cuts a warm rebuild from tens of "
        "seconds to a few. The cache is ORT-version/OS/hardware-specific — a local speedup, not a portable "
        "artifact. Requires 'MLProgram' format.",
    )

    def to_ort(self) -> OrtProvider:
        """Compile to an ONNX Runtime provider tuple.

        Returns:
            The ``CoreMLExecutionProvider`` with format, compute-unit and (optional) cache options.
        """
        options: dict[str, object] = {"ModelFormat": self.model_format, "MLComputeUnits": self.compute_units}
        if self.cache_dir is not None:
            options["ModelCacheDirectory"] = str(self.cache_dir)
        return ("CoreMLExecutionProvider", options)


#: An execution-provider config, discriminated by its ``kind`` tag.
ExecutionProvider = Annotated[
    CpuProvider | CudaProvider | TensorRTProvider | CoreMLProvider,
    Field(discriminator="kind"),
]


class SessionOptionsConfig(BaseConfig):
    """A subset of ONNX Runtime ``SessionOptions`` exposed as typed config."""

    graph_optimization_level: Literal["DISABLE_ALL", "ENABLE_BASIC", "ENABLE_EXTENDED", "ENABLE_ALL"] = Field(
        default="ENABLE_ALL",
        description="Graph optimization level applied when loading the model.",
    )
    intra_op_num_threads: int | None = Field(
        default=None,
        ge=0,
        description="Threads used within a single operator. None uses the ONNX Runtime default.",
    )
    inter_op_num_threads: int | None = Field(
        default=None,
        ge=0,
        description="Threads used across operators. None uses the ONNX Runtime default.",
    )


__all__ = [
    "CoreMLProvider",
    "CpuProvider",
    "CudaProvider",
    "ExecutionProvider",
    "SessionOptionsConfig",
    "TensorRTProvider",
]
