# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# pyright: reportUnknownMemberType=false, reportFunctionMemberAccess=false
# Celery's task `.delay` and config object are not fully typed; silence that noise.

"""Eager (no-broker) CLI entrypoints for the Celery example.

Run with ``python -m celery_app.run train`` or ``python -m celery_app.run forecast``. These
execute the tasks in-process — no broker or worker required — which is the simplest way to
try the example. To exercise the real queue instead, start a worker
(``celery -A celery_app.app worker --pool solo``) and watch it with the Flower UI.
"""

from __future__ import annotations

import logging
import sys

from common.config import Settings

from celery_app import tasks
from celery_app.app import app


def run(action: str) -> None:
    """Run ``train`` or ``forecast`` for every target, in-process (eager mode)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Run tasks inline instead of dispatching to a worker, and keep results in memory so no
    # broker or result backend is required.
    app.conf.update(task_always_eager=True, task_eager_propagates=True, result_backend="cache+memory://")

    task = tasks.train_target if action == "train" else tasks.forecast_target
    for target in Settings().targets:
        task.delay(target).get()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "forecast")
