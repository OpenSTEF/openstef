<!--
SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# OpenSTEF deployment examples

Runnable, self-contained examples showing how to operate OpenSTEF on three popular
orchestrators. They implement the patterns described in the
[deployment guide](../../docs/source/user_guide/guides/deployment.rst):

| Example | Pattern | Best for |
| --- | --- | --- |
| [`dagster_app`](src/dagster_app) | DAG-based orchestration | Teams wanting retries, dependency tracking, audit trails |
| [`airflow_app`](src/airflow_app) | DAG-based orchestration | Teams already running Airflow |
| [`celery_app`](src/celery_app) | Queued execution | Large fleets of forecast targets (fan-out) |

All three share one thin layer in [`common`](src/common):

- **`config.py`** — a single `Settings` object (pydantic-settings) that embeds OpenSTEF's
  own `ForecastingWorkflowConfig`, so every knob lives in one place and is environment
  overridable.
- **`services.py`** — the mocked external systems a real deployment owns: fetching
  measurements from a metering system, fetching a weather forecast from a weather provider,
  and publishing the forecast downstream. They speak OpenSTEF's `TimeSeriesDataset` /
  `ForecastDataset` types — **replace these with your own integrations**.
- **`pipeline.py`** — the real OpenSTEF code path: it combines the fetched data into the
  dataset OpenSTEF expects and builds the workflow. The `fit` / `predict` calls stay in each
  orchestrator.

## Simulated data

Like the [tutorials](../tutorials), these examples load the
[Liander 2024 benchmark dataset](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark)
from the HuggingFace Hub instead of wiring real data sources, so they run end-to-end with
zero external infrastructure. A fixed `reference_time` inside the 2024 data plays the role
of "now".

## Cross-process model handoff

Training and prediction run as **separate processes** in every orchestrator. The training
task persists its model to a local MLflow store (a self-contained SQLite tracking backend
under the data directory); the prediction task creates a fresh workflow and OpenSTEF's
`MLFlowStorageCallback` automatically loads the latest stored model for the same `model_id`.
This is the production-correct pattern — run training at least once before prediction. The
Dagster example shows this between two assets; Airflow and Celery between two tasks.

## Install

From the repository root (installs all three orchestrators plus the `poe` runner):

```bash
uv sync
```

## Run

Each example is wrapped in `poe` tasks so you do not need to know the framework CLIs. Every
framework has the **same three commands** — a UI plus train and forecast — and all of them run
locally with **no external infrastructure** (no database, broker, or message queue):

| Framework | UI | Train (CLI) | Forecast (CLI) |
| --- | --- | --- | --- |
| Dagster | `uv run poe deploy-dagster-ui` | `uv run poe deploy-dagster-train` | `uv run poe deploy-dagster-forecast` |
| Airflow | `uv run poe deploy-airflow-ui` | `uv run poe deploy-airflow-train` | `uv run poe deploy-airflow-forecast` |
| Celery | — | `uv run poe deploy-celery-train` | `uv run poe deploy-celery-forecast` |

The web UIs serve at http://localhost:3000 (Dagster) and :8080 (Airflow).
Run **train before forecast** — the forecast loads the model training persisted.

### Celery has no broker dependency

The Celery example defaults to a **filesystem broker** under the data directory, so a worker
and beat schedule run with no server. The `-train` / `-forecast` tasks run eagerly in-process
(simplest). To exercise the real queue, start a worker:

```bash
uv run --extra celery celery -A celery_app.app worker --pool solo
```

For production scale, point the broker at Redis with no code change:

```bash
export OPENSTEF_DEPLOY_BROKER_URL=redis://localhost:6379/0
export OPENSTEF_DEPLOY_RESULT_BACKEND=redis://localhost:6379/1
```

See each subpackage's module docstring for more, including the real Redis broker, Celery beat,
and the Airflow/Dagster schedulers.
