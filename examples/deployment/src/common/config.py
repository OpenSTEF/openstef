# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Shared configuration for the OpenSTEF deployment examples.

A single :class:`Settings` object holds everything the Dagster, Airflow, and Celery
examples need. Rather than re-declaring forecasting knobs, it *embeds* OpenSTEF's own
:class:`~openstef_models.presets.ForecastingWorkflowConfig` as a field and derives a
per-target copy with :meth:`Settings.config_for`. All fields are environment overridable
with the ``OPENSTEF_DEPLOY_`` prefix.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from openstef_core.types import LeadTime, Q
from openstef_models.integrations.mlflow import MLFlowStorage
from openstef_models.presets import ForecastingWorkflowConfig
from openstef_models.presets.forecasting_workflow import GBLinearHyperParams


def model_id_for(target: str) -> str:
    """Turn a dataset target (e.g. ``mv_feeder/OS Gorredijk``) into a safe model id.

    The id keys the MLflow store and is used in artifact paths, so it must avoid slashes
    and spaces.

    >>> model_id_for("mv_feeder/OS Gorredijk")
    'mv_feeder_OS_Gorredijk'
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", target).strip("_")


class Settings(BaseSettings):
    """Deployment-wide configuration, overridable via ``OPENSTEF_DEPLOY_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="OPENSTEF_DEPLOY_", frozen=True)

    targets: list[str] = Field(
        default=["mv_feeder/OS Gorredijk", "mv_feeder/OS Edam", "mv_feeder/OS Sneek"],
        description="Forecast targets (Liander feeder ids) to train and predict for.",
    )
    data_dir: Path = Field(
        default=Path("openstef_deployment_runs"),
        description="Base directory for the local MLflow store and published forecasts.",
    )
    reference_time: datetime = Field(
        default=datetime(2024, 4, 15, tzinfo=UTC),
        description="Simulated 'now'. A fixed timestamp inside the 2024 benchmark dataset.",
    )
    train_window: timedelta = Field(
        default=timedelta(days=45),
        description="How much history to train on, ending at the reference time.",
    )
    forecasting: ForecastingWorkflowConfig = Field(
        # GBLinear (fast, extrapolates well) with the feature columns named as in the
        # Liander dataset. model_id and mlflow_storage are filled in per target by config_for.
        default=ForecastingWorkflowConfig(
            model_id="openstef-deployment",
            model="gblinear",
            quantiles=[Q(0.1), Q(0.5), Q(0.9)],
            horizons=[LeadTime.from_string("PT36H")],
            target_column="load",
            temperature_column="temperature_2m",
            relative_humidity_column="relative_humidity_2m",
            wind_speed_column="wind_speed_10m",
            radiation_column="shortwave_radiation",
            pressure_column="surface_pressure",
            gblinear_hyperparams=GBLinearHyperParams(n_steps=50),
            mlflow_storage=None,
            verbosity=0,
        ),
        description="Baseline OpenSTEF forecasting config; copied per target.",
    )

    @property
    def output_dir(self) -> Path:
        """Directory where published forecast files are written."""
        return self.data_dir / "forecasts"

    @property
    def dataset_dir(self) -> Path:
        """Directory the (simulated) benchmark dataset is downloaded to."""
        return self.data_dir / "liander_dataset"

    @property
    def horizon(self) -> LeadTime:
        """The (single) forecast horizon used by the examples."""
        return self.forecasting.horizons[0]

    def config_for(self, target: str) -> ForecastingWorkflowConfig:
        """Per-target forecasting config: a copy with the target's model id and shared store."""
        # A local SQLite tracking backend (no server, no deprecation warnings) with an explicit
        # absolute artifact location, so everything stays self-contained under the data directory
        # and the prediction process can load what training persisted.
        data = self.data_dir.resolve()
        return self.forecasting.model_copy(
            update={
                "model_id": model_id_for(target),
                "mlflow_storage": MLFlowStorage(
                    tracking_uri=f"sqlite:///{data / 'mlflow.db'}",
                    local_artifacts_path=data / "mlflow_artifacts",
                    artifact_location=(data / "mlflow_store").as_uri(),
                ),
            },
        )
