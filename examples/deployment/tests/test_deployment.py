# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Network-free tests for the deployment examples' shared layer.

These never download the dataset or train; they pin the configuration wiring that the
Dagster, Airflow, and Celery examples all rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from common import pipeline
from common.config import Settings, model_id_for


def test_model_id_for_is_path_safe() -> None:
    assert model_id_for("mv_feeder/OS Gorredijk") == "mv_feeder_OS_Gorredijk"


def test_config_for_sets_model_id_per_target(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    target = "mv_feeder/OS Gorredijk"

    config = settings.config_for(target)

    assert config.model_id == model_id_for(target)
    assert config.model_id != settings.forecasting.model_id


def test_targets_share_one_mlflow_store(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    first = settings.config_for("mv_feeder/OS Gorredijk")
    second = settings.config_for("mv_feeder/OS Edam")

    assert first.mlflow_storage is not None
    assert second.mlflow_storage is not None
    # Both targets write to the same store (keyed apart by model_id), so the predict
    # process can load what the train process saved.
    assert first.mlflow_storage.local_artifacts_path == second.mlflow_storage.local_artifacts_path


def test_horizon_and_output_dir_derive_from_settings(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    seconds_per_hour = 3600
    horizon_hours = 36

    assert settings.output_dir == tmp_path / "forecasts"
    assert settings.horizon.value.total_seconds() == horizon_hours * seconds_per_hour


def test_targets_override_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSTEF_DEPLOY_TARGETS", '["substation_a", "substation_b"]')

    assert Settings().targets == ["substation_a", "substation_b"]


def test_build_workflow_wires_storage_without_network(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    target = "mv_feeder/OS Gorredijk"

    workflow = pipeline.build_workflow(target, settings=settings)

    assert workflow.model_id == model_id_for(target)
