# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# Airflow's TaskFlow decorators are not fully typed; silence that noise.

"""Airflow training DAG for OpenSTEF (DAG-based orchestration).

Runs daily and trains one model per target, fanning out with Airflow's dynamic task
mapping so each target is an independently retriable task. The trained models are persisted
to the shared MLflow store for the forecast DAG to load.

Run it once without a scheduler with ``uv run poe deploy-airflow-train``.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from common import pipeline
from common.config import Settings

settings = Settings()


@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2024, 4, 1, tz="UTC"),
    catchup=False,
    tags=["openstef"],
)
def openstef_train() -> None:
    """Train one OpenSTEF model per target."""

    @task
    def train(target: str) -> None:
        dataset = pipeline.training_dataset(target, settings=settings)
        workflow = pipeline.build_workflow(target, settings=settings)
        workflow.fit(dataset)

    train.expand(target=settings.targets)


openstef_train()
