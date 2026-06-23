# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the published-checkpoint catalog."""

import pytest

from openstef_foundation_models.models.catalog import CheckpointVariant, Chronos2


def test_dynamic_variant_targets_the_base_repo_and_unsuffixed_file() -> None:
    """The dynamic variant maps to `OpenSTEF/<slug>-onnx` and `<slug>.onnx`."""
    checkpoint = Chronos2.BASE.checkpoint()
    assert checkpoint.repo_id == "OpenSTEF/chronos-2-onnx"
    assert checkpoint.filename == "chronos-2.onnx"


def test_static_variant_adds_the_static_filename_suffix() -> None:
    """The static variant keeps the repo but appends `_static` to the filename."""
    checkpoint = Chronos2.SMALL.checkpoint(CheckpointVariant.STATIC)
    assert checkpoint.repo_id == "OpenSTEF/chronos-2-small-onnx"
    assert checkpoint.filename == "chronos-2-small_static.onnx"


def test_metadata_filename_defaults_alongside_the_weights() -> None:
    """The metadata filename is derived from the weights stem on resolve."""
    checkpoint = Chronos2.BASE.checkpoint(CheckpointVariant.STATIC)
    # metadata_filename defaults to None; HubCheckpoint derives it from the weights stem.
    assert checkpoint.metadata_filename is None


def test_recommended_variant_is_static_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS gets STATIC (so CoreML can engage); other platforms get DYNAMIC."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert CheckpointVariant.recommended() is CheckpointVariant.STATIC
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert CheckpointVariant.recommended() is CheckpointVariant.DYNAMIC
