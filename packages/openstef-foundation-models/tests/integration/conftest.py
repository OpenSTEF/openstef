# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared fixtures for the foundation-model integration tests.

These tests run the real exported Chronos-2 ONNX checkpoint. The artifact is
large and lives outside the repository (the export lab), so every fixture here
skips cleanly when it is unavailable. Point at a custom artifact with the
``OPENSTEF_CHRONOS2_ONNX_PATH`` environment variable.
"""

import os
from pathlib import Path

import pytest

from openstef_foundation_models.inference.onnx_backend import OnnxBackend
from openstef_foundation_models.models.checkpoint import (
    CheckpointMetadata,
    LocalCheckpoint,
    ResolvedCheckpoint,
)

#: Chronos-2's native quantile grid (21 levels), read from ``chronos_config.quantiles``.
NATIVE_QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]  # fmt: skip

#: Sizing of the exported graph (60-day context, 7-day horizon @ 15 min).
CONTEXT_LENGTH = 5760
OUTPUT_PATCH_SIZE = 16
HORIZON_PATCHES = 42  # horizon_length = 672 steps
RESOLUTION_MINUTES = 15

#: Default location of the exported FP32 weights inside the export lab.
_DEFAULT_ARTIFACT = Path("chronos-onnx-lab/artifacts/chronos-2.onnx")
_ARTIFACT_ENV_VAR = "OPENSTEF_CHRONOS2_ONNX_PATH"


def _repo_root() -> Path:
    """Locate the repository root (four levels above this test file)."""
    return Path(__file__).resolve().parents[4]


@pytest.fixture(scope="session")
def onnx_artifact() -> Path:
    """Resolve the Chronos-2 ONNX weights, skipping the test when absent."""
    override = os.environ.get(_ARTIFACT_ENV_VAR)
    path = Path(override) if override else _repo_root() / _DEFAULT_ARTIFACT
    if not path.is_file():
        pytest.skip(f"Chronos-2 ONNX artifact not found at {path}; set {_ARTIFACT_ENV_VAR} to run this test.")
    return path


@pytest.fixture(scope="session")
def chronos2_metadata() -> CheckpointMetadata:
    """Metadata describing the exported Chronos-2 checkpoint."""
    return CheckpointMetadata(
        model_family="chronos2",
        input_names=["context", "group_ids", "attention_mask"],
        output_name="quantile_preds",
        native_quantiles=NATIVE_QUANTILES,
        context_length=CONTEXT_LENGTH,
        output_patch_size=OUTPUT_PATCH_SIZE,
        horizon_patches=HORIZON_PATCHES,
        resolution_minutes=RESOLUTION_MINUTES,
    )


@pytest.fixture(scope="session")
def resolved_checkpoint(
    onnx_artifact: Path,
    chronos2_metadata: CheckpointMetadata,
    tmp_path_factory: pytest.TempPathFactory,
) -> ResolvedCheckpoint:
    """Resolve the artifact against a sidecar metadata file written to a temp dir."""
    metadata_path = tmp_path_factory.mktemp("chronos2-meta") / "chronos-2.metadata.json"
    metadata_path.write_text(chronos2_metadata.model_dump_json(), encoding="utf-8")
    return LocalCheckpoint(path=onnx_artifact, metadata_path=metadata_path).resolve()


@pytest.fixture(scope="session")
def onnx_backend(resolved_checkpoint: ResolvedCheckpoint) -> OnnxBackend:
    """A single ONNX backend, built once and shared across the session (load-once)."""
    return OnnxBackend.from_checkpoint(resolved_checkpoint)
