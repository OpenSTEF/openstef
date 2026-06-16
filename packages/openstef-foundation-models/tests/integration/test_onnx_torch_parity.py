# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""ONNX-vs-Torch parity check for the exported Chronos-2 graph.

Confirms the exported ONNX model reproduces the original Torch module's
quantile predictions. The Torch side reuses the export wrapper
(``Chronos2OnnxModule``) from the export lab so both sides share the exact same
named-tensor contract.

This test is heavy and depends on optional, large dependencies (``torch`` +
``chronos-forecasting``) and the export lab, so it skips unless everything is
available.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from openstef_foundation_models.inference.onnx_backend import OnnxBackend
from openstef_foundation_models.models.checkpoint import CheckpointMetadata

pytestmark = [pytest.mark.slow, pytest.mark.integration]

_LAB_EXPORT_SCRIPT = Path("chronos-onnx-lab/export_chronos2_to_onnx.py")
_CHRONOS_MODEL_ID = "amazon/chronos-2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_lab_module():  # noqa: ANN202  # dynamically loaded module
    """Import the git-untracked export lab module, skipping if unavailable."""
    script = _repo_root() / _LAB_EXPORT_SCRIPT
    if not script.is_file():
        pytest.skip(f"Export lab script not found at {script}.")
    spec = importlib.util.spec_from_file_location("_chronos2_export_lab", script)
    if spec is None or spec.loader is None:
        pytest.skip("Could not load the export lab module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onnx_matches_torch_quantile_predictions(
    onnx_backend: OnnxBackend, chronos2_metadata: CheckpointMetadata
) -> None:
    """ONNX and Torch produce the same quantile predictions for identical inputs."""
    # Arrange
    pytest.importorskip("torch", reason="ONNX-vs-Torch parity needs the [torch] extra.")
    pytest.importorskip("chronos", reason="ONNX-vs-Torch parity needs chronos-forecasting.")
    import torch  # ty: ignore[unresolved-import]  # noqa: PLC0415
    from chronos import BaseChronosPipeline  # ty: ignore[unresolved-import]  # noqa: PLC0415

    lab = _load_lab_module()
    pipeline = BaseChronosPipeline.from_pretrained(_CHRONOS_MODEL_ID, device_map="cpu")
    torch_module = lab.Chronos2OnnxModule(pipeline.model, num_output_patches=chronos2_metadata.horizon_patches).eval()

    batch_size, context_length = 2, 512
    rng = np.random.default_rng(7)
    context = rng.normal(100.0, 10.0, size=(batch_size, context_length)).astype(np.float32)
    group_ids = np.arange(batch_size, dtype=np.int64)
    attention_mask = np.ones((batch_size, context_length), dtype=np.float32)
    # Target-only parity: covariates masked out over the frozen horizon.
    horizon_length = chronos2_metadata.horizon_length
    future_covariates = np.zeros((batch_size, horizon_length), dtype=np.float32)
    future_covariates_mask = np.zeros((batch_size, horizon_length), dtype=np.float32)

    # Act
    onnx_out = np.asarray(
        onnx_backend.run(
            {
                "context": context,
                "group_ids": group_ids,
                "attention_mask": attention_mask,
                "future_covariates": future_covariates,
                "future_covariates_mask": future_covariates_mask,
            }
        )["quantile_preds"]
    )
    with torch.inference_mode():
        torch_out = (
            torch_module(
                torch.from_numpy(context),
                torch.from_numpy(group_ids),
                torch.from_numpy(attention_mask),
                torch.from_numpy(future_covariates),
                torch.from_numpy(future_covariates_mask),
            )
            .cpu()
            .numpy()
        )

    # Assert
    assert onnx_out.shape == torch_out.shape
    np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-3, atol=1e-3)
