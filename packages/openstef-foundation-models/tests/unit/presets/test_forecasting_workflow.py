# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the foundation-model forecasting preset and factory."""

import subprocess  # noqa: S404
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

import openstef_foundation_models.inference.onnx_backend as onnx_module
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.models.checkpoint import (
    CheckpointMetadata,
    LocalCheckpoint,
    ResolvedCheckpoint,
)
from openstef_foundation_models.models.forecasting import Chronos2Forecaster
from openstef_foundation_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    OnnxBackendConfig,
    create_forecasting_workflow,
)
from openstef_models.models import ForecastingModel
from openstef_models.transforms.general import Selector
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

NATIVE_QUANTILES = [0.1, 0.5, 0.9]


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        model_family="chronos2",
        input_names=["context", "group_ids", "attention_mask"],
        output_name="quantile_preds",
        native_quantiles=NATIVE_QUANTILES,
        context_length=64,
        output_patch_size=16,
        horizon_patches=2,
        resolution_minutes=15,
    )


class StubBackend:
    """Minimal :class:`InferenceBackend` used to assert factory wiring."""

    def __init__(self, metadata: CheckpointMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> CheckpointMetadata:
        return self._metadata

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _write_checkpoint(tmp_path: Path) -> LocalCheckpoint:
    """Write a dummy weights file and a valid metadata sidecar to *tmp_path*."""
    weights_path = tmp_path / "chronos-2.onnx"
    weights_path.write_bytes(b"")
    metadata_path = tmp_path / "chronos-2.metadata.json"
    metadata_path.write_text(_metadata().model_dump_json(), encoding="utf-8")
    return LocalCheckpoint(path=weights_path)


def test_create_forecasting_workflow_builds_chronos2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory composes the built backend into a Chronos2Forecaster workflow."""
    # Arrange
    backend = StubBackend(_metadata())
    monkeypatch.setattr(OnnxBackendConfig, "build", lambda _self: backend)
    config = ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=LocalCheckpoint(path=Path("chronos-2.onnx")),
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime.from_string("PT24H")],
    )

    # Act
    workflow = create_forecasting_workflow(config)

    # Assert
    assert isinstance(workflow, CustomForecastingWorkflow)
    assert workflow.model_id == "chronos2"
    model = workflow.model
    assert isinstance(model, ForecastingModel)
    assert model.target_column == "load"
    forecaster = model.forecaster
    assert isinstance(forecaster, Chronos2Forecaster)
    assert forecaster.backend is backend
    assert forecaster.quantiles == [Quantile(0.1), Quantile(0.5), Quantile(0.9)]
    assert forecaster.horizons == [LeadTime.from_string("PT24H")]


def test_create_forecasting_workflow_selects_target_and_weather_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no explicit selection, the target plus the three weather covariates are kept."""
    # Arrange
    monkeypatch.setattr(OnnxBackendConfig, "build", lambda _self: StubBackend(_metadata()))
    config = ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=LocalCheckpoint(path=Path("chronos-2.onnx")),
    )

    # Act
    workflow = create_forecasting_workflow(config)

    # Assert
    selector = workflow.model.preprocessing.transforms[0]
    assert isinstance(selector, Selector)
    assert selector.selection.include == {
        "load",
        "shortwave_radiation",
        "wind_speed_80m",
        "temperature_2m",
    }


def test_onnx_backend_config_build_resolves_checkpoint_and_passes_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build() resolves the checkpoint and forwards provider options to the backend."""
    # Arrange
    captured: dict[str, object] = {}

    def fake_from_checkpoint(
        resolved: ResolvedCheckpoint,
        providers: object = None,
        session_options: object = None,
        *,
        strict_providers: bool = False,
    ) -> StubBackend:
        captured["resolved"] = resolved
        captured["strict_providers"] = strict_providers
        return StubBackend(resolved.metadata)

    monkeypatch.setattr(onnx_module.OnnxBackend, "from_checkpoint", staticmethod(fake_from_checkpoint))
    backend_config = OnnxBackendConfig(checkpoint=_write_checkpoint(tmp_path), strict_providers=True)

    # Act
    backend = backend_config.build()

    # Assert
    assert isinstance(backend, StubBackend)
    resolved = captured["resolved"]
    assert isinstance(resolved, ResolvedCheckpoint)
    assert resolved.metadata.model_family == "chronos2"
    assert captured["strict_providers"] is True


def test_config_round_trips_through_json() -> None:
    """The config serialises and validates back to an equal config."""
    # Arrange
    config = ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=LocalCheckpoint(path=Path("chronos-2.onnx")),
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime.from_string("PT48H")],
    )

    # Act
    restored = ForecastingWorkflowConfig.model_validate_json(config.model_dump_json())

    # Assert
    assert restored == config


def test_importing_preset_does_not_import_onnxruntime() -> None:
    """Importing the preset must not eagerly import the ONNX Runtime dependency."""
    # Arrange / Act: a fresh interpreter so prior test imports cannot leak in
    code = (
        "import sys; import openstef_foundation_models.presets.forecasting_workflow; "
        "print('onnxruntime' in sys.modules)"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert
    assert result.stdout.strip() == "False"
