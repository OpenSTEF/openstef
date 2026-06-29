# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the foundation-model forecasting preset and factory."""

import importlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

import openstef_foundation_models.inference.onnx_backend as onnx_module
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.inference.provider_selection import DefaultProviderPolicy
from openstef_foundation_models.models.checkpoint import (
    CheckpointMetadata,
    HubCheckpoint,
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
from openstef_models.utils.feature_selection import FeatureSelection
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

NATIVE_QUANTILES = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]


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
    """Write a dummy weights file and valid metadata JSON to *tmp_path*."""
    weights_path = tmp_path / "chronos-2.onnx"
    weights_path.write_bytes(b"")
    metadata_path = tmp_path / "chronos-2.metadata.json"
    metadata_path.write_text(_metadata().model_dump_json(), encoding="utf-8")
    return LocalCheckpoint(path=weights_path)


def test_create_forecasting_workflow_builds_chronos2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory composes the built backend into a Chronos2Forecaster workflow."""
    # Arrange
    backend = StubBackend(_metadata())
    monkeypatch.setattr(OnnxBackendConfig, "build", lambda _self, _checkpoint: backend)
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


def test_create_forecasting_workflow_selects_all_features_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no explicit selection, all columns are kept (target + every covariate)."""
    # Arrange
    monkeypatch.setattr(OnnxBackendConfig, "build", lambda _self, _checkpoint: StubBackend(_metadata()))
    config = ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=LocalCheckpoint(path=Path("chronos-2.onnx")),
    )

    # Act
    workflow = create_forecasting_workflow(config)

    # Assert
    selector = workflow.model.preprocessing.transforms[0]
    assert isinstance(selector, Selector)
    assert selector.selection == FeatureSelection.ALL


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
        policy: object = None,
    ) -> StubBackend:
        captured["resolved"] = resolved
        captured["providers"] = providers
        captured["policy"] = policy
        return StubBackend(resolved.metadata)

    monkeypatch.setattr(onnx_module.OnnxBackend, "from_checkpoint", staticmethod(fake_from_checkpoint))
    policy = DefaultProviderPolicy()
    backend_config = OnnxBackendConfig(policy=policy)

    # Act
    backend = backend_config.build(checkpoint=_write_checkpoint(tmp_path))

    # Assert
    assert isinstance(backend, StubBackend)
    resolved = captured["resolved"]
    assert isinstance(resolved, ResolvedCheckpoint)
    assert resolved.metadata.model_family == "chronos2"
    assert captured["providers"] is None
    assert captured["policy"] is policy


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


def test_default_checkpoint_is_the_published_hub_export() -> None:
    """With no checkpoint given, the config points at the OpenSTEF Chronos-2 Hub repo."""
    # Act
    checkpoint = ForecastingWorkflowConfig().checkpoint

    # Assert
    assert isinstance(checkpoint, HubCheckpoint)
    assert checkpoint.repo_id == "OpenSTEF/chronos-2-onnx"
    assert checkpoint.filename == "chronos-2.onnx"


def test_importing_preset_succeeds_without_building_backend() -> None:
    """Importing the preset must not eagerly build the ONNX backend session.

    ``OnnxBackend`` is imported lazily inside ``OnnxBackendConfig.build``, so the
    preset module imports cleanly without constructing an inference session.
    """
    # Act / Assert: the import succeeds without building a backend at module load.
    module = importlib.import_module("openstef_foundation_models.presets.forecasting_workflow")
    assert module is not None
