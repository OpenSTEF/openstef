# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# Airflow's TaskFlow decorators are not fully typed; silence that noise.

"""Airflow forecast DAG for OpenSTEF (DAG-based orchestration).

Runs hourly and, for each target, assembles prediction input, predicts, and publishes the
result. Each forecast process loads the model the training DAG persisted to the shared
MLflow store, so run ``openstef_train`` at least once first.

This example uses a fixed ``reference_time`` (inside the 2024 benchmark data) as "now". In a
real deployment you would instead derive it from the run's logical date
(``context["data_interval_end"]``).

Run it once without a scheduler with ``uv run poe deploy-airflow-forecast``.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from common import pipeline, services
from common.config import Settings

settings = Settings()


@dag(
    schedule="@hourly",
    start_date=pendulum.datetime(2024, 4, 1, tz="UTC"),
    catchup=False,
    tags=["openstef"],
)
def openstef_forecast() -> None:
    """Forecast and publish for every target."""

    @task
    def forecast(target: str) -> None:
        dataset = pipeline.prediction_dataset(target, settings=settings)
        workflow = pipeline.build_workflow(target, settings=settings)
        result = workflow.predict(dataset, forecast_start=settings.reference_time)
        services.publish_forecast(result, target, settings=settings)

    forecast.expand(target=settings.targets)


openstef_forecast()
