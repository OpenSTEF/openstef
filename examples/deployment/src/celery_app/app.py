# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# pyright: reportUnknownMemberType=false
# Celery's config object is dynamically typed; silence the resulting "unknown member"
# noise. Real type checking stays on for everything else in this file.

"""Celery application for the OpenSTEF queued-execution example.

**Local by default, no server required.** The broker and result backend default to a
filesystem queue under the data directory, so a worker, beat schedule, and the Flower UI all
run on your machine out of the box. For production scale, point the broker at Redis (or
another broker) — no code change, just environment variables::

    export OPENSTEF_DEPLOY_BROKER_URL=redis://localhost:6379/0
    export OPENSTEF_DEPLOY_RESULT_BACKEND=redis://localhost:6379/1

Run it::

    uv run poe deploy-celery-train      # train every target (eager, in-process, no worker)
    uv run poe deploy-celery-forecast   # forecast every target (eager)
    uv run poe deploy-celery-ui         # the Flower monitoring UI

    # the real queue: a worker pulling from the local filesystem broker
    uv run --extra celery celery -A celery_app.app worker --pool solo
"""

from __future__ import annotations

import os
from pathlib import Path

from celery import Celery
from celery.schedules import crontab

# Filesystem broker + file result backend need no server. Kombu requires the queue folders
# to exist up front. (Ignored when a real broker URL is provided via the environment.)
_queue_dir = Path(os.environ.get("OPENSTEF_DEPLOY_DATA_DIR", "openstef_deployment_runs")) / "celery"
for _sub in ("broker", "control", "results"):
    (_queue_dir / _sub).mkdir(parents=True, exist_ok=True)

app = Celery(
    "openstef_deployment",
    broker=os.environ.get("OPENSTEF_DEPLOY_BROKER_URL", "filesystem://"),
    backend=os.environ.get("OPENSTEF_DEPLOY_RESULT_BACKEND", f"file://{_queue_dir / 'results'}"),
    include=["celery_app.tasks"],
)

app.conf.update(
    timezone="UTC",
    broker_transport_options={
        "data_folder_in": str(_queue_dir / "broker"),
        "data_folder_out": str(_queue_dir / "broker"),
        # Keep the pidbox/control exchange under the data dir too (else Kombu writes
        # a ./control/ folder relative to the working directory).
        "control_folder": str(_queue_dir / "control"),
    },
    beat_schedule={
        "retrain-daily": {"task": "celery_app.tasks.train_all", "schedule": crontab(hour="2", minute="0")},
        "forecast-hourly": {"task": "celery_app.tasks.forecast_all", "schedule": crontab(minute="0")},
    },
)
