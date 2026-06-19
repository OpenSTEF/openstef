# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared fixtures for the foundation-model integration tests.

These tests run the real exported Chronos-2 ONNX checkpoint. The artifact is
large and lives outside the repository, so every fixture here skips cleanly when
it is unavailable. The checkpoint metadata is read from the ``.metadata.json``
file written next to the weights.
"""

from pathlib import Path

import pytest

from openstef_foundation_models.inference.onnx_backend import OnnxBackend
from openstef_foundation_models.models.checkpoint import (
    CheckpointMetadata,
    LocalCheckpoint,
    ResolvedCheckpoint,
)

#: Default location of the exported FP32 weights inside the export lab.
_DEFAULT_ARTIFACT = Path("chronos-onnx-lab/artifacts/chronos-2.onnx")


def _repo_root() -> Path:
    """Locate the repository root (four levels above this test file)."""
    return Path(__file__).resolve().parents[4]


@pytest.fixture(scope="session")
def onnx_artifact() -> Path:
    """Resolve the Chronos-2 ONNX weights, skipping the test when absent."""
    path = _repo_root() / _DEFAULT_ARTIFACT
    if not path.is_file():
        pytest.skip(f"Chronos-2 ONNX artifact not found at {path}; export it with the chronos-onnx-lab script.")
    return path


@pytest.fixture(scope="session")
def chronos2_metadata(onnx_artifact: Path) -> CheckpointMetadata:
    """Metadata describing the exported checkpoint, read from its JSON file."""
    metadata_path = onnx_artifact.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        pytest.skip(f"Checkpoint metadata not found at {metadata_path}; re-export to generate it.")
    return CheckpointMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def resolved_checkpoint(onnx_artifact: Path) -> ResolvedCheckpoint:
    """Resolve the artifact against its auto-discovered metadata file."""
    return LocalCheckpoint(path=onnx_artifact).resolve()


@pytest.fixture(scope="session")
def onnx_backend(resolved_checkpoint: ResolvedCheckpoint) -> OnnxBackend:
    """A single ONNX backend, built once and shared across the session (load-once)."""
    return OnnxBackend.from_checkpoint(resolved_checkpoint)
