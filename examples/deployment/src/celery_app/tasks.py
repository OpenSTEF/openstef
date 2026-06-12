# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


# Celery's task decorator returns a loosely typed object, so `.s` / `.delay` and the
# config are not statically known. Disable that noise; real type checking stays on.

"""Celery tasks for the OpenSTEF queued-execution example.

Each target is a lightweight per-location task dispatched to the worker pool — the pattern
that scales to thousands of grid connections. ``train_all`` / ``forecast_all`` fan the work
out as a Celery ``group``; the beat schedule in :mod:`celery_app.app` drives them.

The forecast task loads the model the training task persisted to the shared MLflow store, so
train before forecasting.
"""

from __future__ import annotations

from celery import group
from common import pipeline, services
from common.config import Settings

from celery_app.app import app

settings = Settings()


@app.task
def train_target(target: str) -> None:
    """Train and persist a model for one target."""
    dataset = pipeline.training_dataset(target, settings=settings)
    workflow = pipeline.build_workflow(target, settings=settings)
    workflow.fit(dataset)


@app.task
def forecast_target(target: str) -> str:
    """Forecast one target, publish the result, and return its path."""
    dataset = pipeline.prediction_dataset(target, settings=settings)
    workflow = pipeline.build_workflow(target, settings=settings)
    forecast = workflow.predict(dataset, forecast_start=settings.reference_time)
    return str(services.publish_forecast(forecast, target, settings=settings))


@app.task
def train_all() -> None:
    """Fan out a training task per target."""
    group(train_target.s(target) for target in settings.targets).apply_async()


@app.task
def forecast_all() -> None:
    """Fan out a forecast task per target."""
    group(forecast_target.s(target) for target in settings.targets).apply_async()
