# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the ONNX backend's provider-selection and fallback behaviour.

These exercise the strict-vs-graceful contract through the public
:meth:`OnnxBackend.from_checkpoint` seam, with a faked ONNX Runtime session: an
explicitly requested chain is enforced strictly (a missing accelerator raises),
while a policy-selected chain is graceful (it warns only on a full fallback to
CPU).
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import pytest

import openstef_foundation_models.inference.onnx_backend as onnx_module
from openstef_core.types import Quantile
from openstef_foundation_models.inference.onnx_backend import OnnxBackend
from openstef_foundation_models.inference.provider_selection import HostCapabilities
from openstef_foundation_models.inference.providers import (
    CoreMLProvider,
    CpuProvider,
    CudaProvider,
    ExecutionProvider,
)
from openstef_foundation_models.models.checkpoint import CheckpointMetadata, ResolvedCheckpoint

CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"


def _checkpoint() -> ResolvedCheckpoint:
    metadata = CheckpointMetadata(
        model_family="chronos2",
        input_names=["context"],
        output_name="quantile_preds",
        native_quantiles=[Quantile(0.5)],
        context_length=64,
        output_patch_size=16,
        horizon_patches=2,
        resolution_minutes=15,
    )
    return ResolvedCheckpoint(weights_path=Path("model.onnx"), metadata=metadata)


class _FixedPolicy:
    """A policy that returns a preset chain regardless of checkpoint or host."""

    def __init__(self, chain: list[ExecutionProvider]) -> None:
        self._chain = chain

    def select(self, metadata: CheckpointMetadata, host: HostCapabilities) -> list[ExecutionProvider]:
        return self._chain


def _patch_session(monkeypatch: pytest.MonkeyPatch, realized: Iterable[str]) -> None:
    """Replace ``ort.InferenceSession`` with a fake reporting *realized* providers."""
    realized_list = list(realized)

    class _FakeSession:
        def __init__(self, path: str, sess_options: object = None, providers: object = None) -> None:
            self.requested = providers

        def get_providers(self) -> list[str]:
            return realized_list

    monkeypatch.setattr(onnx_module.ort, "InferenceSession", _FakeSession)


def test_explicit_chain_raises_when_accelerator_drops_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit accelerator that ONNX Runtime drops to CPU is a strict error."""
    _patch_session(monkeypatch, [CPU])
    with pytest.raises(RuntimeError, match=CUDA):
        OnnxBackend.from_checkpoint(_checkpoint(), providers=[CudaProvider(), CpuProvider()])


def test_explicit_chain_accepts_realized_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A strict chain whose requested accelerator was realized builds cleanly."""
    _patch_session(monkeypatch, [CUDA, CPU])
    backend = OnnxBackend.from_checkpoint(_checkpoint(), providers=[CudaProvider(), CpuProvider()])
    assert backend.metadata.model_family == "chronos2"


def test_policy_chain_warns_on_full_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A policy chain that ends up entirely on CPU is worth a warning, not an error."""
    _patch_session(monkeypatch, [CPU])
    with caplog.at_level(logging.WARNING):
        OnnxBackend.from_checkpoint(_checkpoint(), policy=_FixedPolicy([CoreMLProvider(), CpuProvider()]))
    assert "fell back" in caplog.text


def test_policy_chain_silent_when_accelerator_realized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A policy chain realizing its accelerator is the intended outcome, not a reported fallback."""
    _patch_session(monkeypatch, [COREML, CPU])
    with caplog.at_level(logging.WARNING):
        OnnxBackend.from_checkpoint(_checkpoint(), policy=_FixedPolicy([CoreMLProvider(), CpuProvider()]))
    assert caplog.records == []
