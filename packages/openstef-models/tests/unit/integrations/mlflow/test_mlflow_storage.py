# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from openstef_core.mixins import HyperParams, Stateful
from openstef_models.integrations.mlflow import MLFlowStorage

if TYPE_CHECKING:
    from pathlib import Path


class SimpleHyperParams(HyperParams):
    """Simple hyperparameters for testing."""

    learning_rate: float = 0.01
    max_depth: int = 5


class SimpleStatefulModel(Stateful):
    """Simple model for testing serialization."""

    def __init__(self) -> None:
        self.param_a: str = "initial"
        self.param_b: int = 42


@pytest.fixture
def storage(tmp_path: Path) -> MLFlowStorage:
    """Create MLflow storage instance with temporary directories."""
    tracking_path = tmp_path / "mlflow"
    artifacts_path = tmp_path / "mlflow_artifacts"

    return MLFlowStorage(
        tracking_uri=str(tracking_path),
        local_artifacts_path=artifacts_path,
    )


@pytest.fixture
def model_id() -> str:
    """Return consistent model identifier for tests."""
    return "test_model_123"


def test_create_run(storage: MLFlowStorage, model_id: str):
    """Test that create_run converts HyperParams to MLflow parameters."""
    # Arrange
    hyperparams = SimpleHyperParams(learning_rate=0.05, max_depth=10)

    # Act
    run = storage.create_run(model_id=model_id, hyperparams=hyperparams)
    run_id = cast(str, run.info.run_id)

    # Assert
    fetched_run = storage._client.get_run(run_id)
    assert fetched_run.data.params["learning_rate"] == "0.05"  # type: ignore
    assert fetched_run.data.params["max_depth"] == "10"  # type: ignore


def test_create_run__experiment_prefix(tmp_path: Path, model_id: str):
    """Test that experiment_name_prefix is prepended to experiment names."""
    # Arrange
    storage = MLFlowStorage(
        tracking_uri=str(tmp_path / "mlflow"),
        local_artifacts_path=tmp_path / "artifacts",
        experiment_name_prefix="prod_",
    )

    # Act
    run = storage.create_run(model_id=model_id)
    experiment_id = cast(str, run.info.experiment_id)  # type: ignore

    # Assert
    experiment = storage._client.get_experiment(experiment_id)
    assert experiment.name == f"prod_{model_id}"  # type: ignore


def test_create_run__reuses_experiment(storage: MLFlowStorage, model_id: str):
    """Test that multiple runs for same model_id share the same experiment."""
    # Arrange
    first_run = storage.create_run(model_id=model_id)
    first_exp_id = cast(str, first_run.info.experiment_id)  # type: ignore

    # Act
    second_run = storage.create_run(model_id=model_id)
    second_exp_id = cast(str, second_run.info.experiment_id)  # type: ignore

    # Assert
    assert first_exp_id == second_exp_id


def test_model_roundtrip(storage: MLFlowStorage, model_id: str):
    """Test save and load roundtrip preserves model state."""
    # Arrange
    run = storage.create_run(model_id=model_id)
    run_id = cast(str, run.info.run_id)
    original_model = SimpleStatefulModel()
    original_model.param_a = "trained_value"
    original_model.param_b = 99

    # Act
    storage.save_run_model(model_id=model_id, run_id=run_id, model=original_model)
    storage.finalize_run(model_id=model_id, run_id=run_id)
    loaded_model = storage.load_run_model(run_id=run_id)

    # Assert
    assert isinstance(loaded_model, SimpleStatefulModel)
    assert loaded_model.param_a == "trained_value"
    assert loaded_model.param_b == 99


def test_search_latest_runs(storage: MLFlowStorage, model_id: str):
    """Test that search_latest_runs returns only the most recent run."""
    # Arrange - Create multiple runs
    first_run = storage.create_run(model_id=model_id)
    storage.finalize_run(model_id=model_id, run_id=cast(str, first_run.info.run_id))

    second_run = storage.create_run(model_id=model_id)
    storage.finalize_run(model_id=model_id, run_id=cast(str, second_run.info.run_id))

    third_run = storage.create_run(model_id=model_id)
    third_run_id = cast(str, third_run.info.run_id)
    storage.finalize_run(model_id=model_id, run_id=third_run_id)

    # Act
    latest_runs = storage.search_latest_runs(model_id=model_id, limit=1)

    # Assert - Should return only the most recent run
    assert len(latest_runs) == 1
    assert cast(str, latest_runs[0].info.run_id) == third_run_id


def test_search_latest_runs__no_experiment(storage: MLFlowStorage):
    """Test that search_latest_runs handles non-existent experiments."""
    # Arrange - No runs created for this model_id

    # Act
    latest_runs = storage.search_latest_runs(model_id="nonexistent_model", limit=10)

    # Assert
    assert latest_runs == []
