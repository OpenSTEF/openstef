Production Deployment
=====================

This page covers patterns for deploying OpenSTEF forecasting pipelines in production environments. Whether you run a simple cron job on a single server or orchestrate hundreds of forecast targets across a cloud platform, OpenSTEF's library design gives you full control over how pipelines are scheduled, containerized, and monitored.

For data source configuration, see :doc:`data_integration`. For use-case-specific patterns, see :doc:`use_cases`.

.. mermaid:: /diagrams/user_guide/deployment_diagram_1.mmd

Deployment Approaches
---------------------

OpenSTEF is a library, not a service. You embed it in your own application and choose how to trigger execution. The three most common patterns are:

- **Scheduled scripts** — cron or systemd timers calling a Python entrypoint
- **Task queues** — Celery, Airflow, or Dagster orchestrating train/predict tasks
- **Serverless functions** — AWS Lambda or Azure Functions triggered on a schedule

All patterns share the same core: configure a workflow, call ``fit()`` on a schedule (e.g., daily), and call ``predict()`` at forecast cadence (e.g., every 15 minutes).

Minimal Production Script
-------------------------

A production entrypoint typically loads configuration, builds the workflow, fetches data, and runs prediction:

.. code-block:: python

   import logging
   from datetime import timedelta
   from pathlib import Path

   from openstef_core.datasets import ForecastDataset
   from openstef_core.types import LeadTime, Q
   from openstef_models.integrations.mlflow import MLFlowStorage
   from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow

   logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
   logger = logging.getLogger(__name__)

   def create_workflow(model_dir: Path):
       return create_forecasting_workflow(
           config=ForecastingWorkflowConfig(
               model_id="production_forecaster_v1",
               model="gblinear",
               horizons=[LeadTime.from_string("PT36H")],
               quantiles=[Q(0.5), Q(0.1), Q(0.9)],
               mlflow_storage=MLFlowStorage(
                   tracking_uri=str(model_dir / "mlflow_tracking"),
                   local_artifacts_path=model_dir / "mlflow_artifacts",
               ),
           )
       )

   def run_predict(workflow, dataset):
       """Run prediction and return forecast."""
       forecast: ForecastDataset = workflow.predict(dataset)
       logger.info("Forecast generated: %d rows", len(forecast.data))
       return forecast

   def run_train(workflow, dataset):
       """Retrain model on latest data."""
       result = workflow.fit(dataset)
       if result is not None:
           logger.info("Training complete. Metrics:\n%s", result.metrics_full.to_dataframe())
       return result

Containerization
----------------

Package your forecasting application in a Docker container for reproducible deployments:

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY src/ ./src/
   COPY configs/ ./configs/

   # Model artifacts volume
   VOLUME /models

   ENV MODEL_DIR=/models
   ENV PYTHONUNBUFFERED=1

   ENTRYPOINT ["python", "-m", "src.main"]

Your ``requirements.txt`` should pin OpenSTEF packages:

.. code-block:: text

   openstef-core>=4.0,<5.0
   openstef-models>=4.0,<5.0
   openstef-beam>=4.0,<5.0

.. warning::

   Always pin major versions. OpenSTEF V4 has breaking changes from V3 — see :doc:`migration_v3_v4` for details.

Scheduling with Cron
---------------------

The simplest production deployment uses cron to trigger training and prediction separately:

.. code-block:: bash

   # /etc/cron.d/openstef-forecasting

   # Retrain models daily at 02:00 UTC
   0 2 * * * appuser /app/venv/bin/python -m src.main --mode train

   # Generate forecasts every 15 minutes
   */15 * * * * appuser /app/venv/bin/python -m src.main --mode predict

Structure your entrypoint to accept a mode argument:

.. code-block:: python

   import argparse
   from pathlib import Path

   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--mode", choices=["train", "predict"], required=True)
       parser.add_argument("--model-dir", type=Path, default=Path("/models"))
       args = parser.parse_args()

       workflow = create_workflow(args.model_dir)
       dataset = load_latest_data()  # Your data loading logic

       if args.mode == "train":
           run_train(workflow, dataset)
       else:
           run_predict(workflow, dataset)

   if __name__ == "__main__":
       main()

Orchestration with Airflow
--------------------------

For multi-target deployments, Apache Airflow provides dependency management, retries, and monitoring:

.. code-block:: python

   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime, timedelta

   default_args = {
       "retries": 3,
       "retry_delay": timedelta(minutes=5),
       "execution_timeout": timedelta(minutes=30),
   }

   with DAG(
       "openstef_forecast",
       default_args=default_args,
       schedule_interval="*/15 * * * *",
       start_date=datetime(2024, 1, 1),
       catchup=False,
   ) as dag:

       def predict_target(target_id: str, **kwargs):
           from src.forecasting import create_workflow, load_data, publish_forecast
           from pathlib import Path

           workflow = create_workflow(Path(f"/models/{target_id}"))
           dataset = load_data(target_id)
           forecast = workflow.predict(dataset)
           publish_forecast(target_id, forecast)

       # Create a task per forecast target
       target_ids = ["substation_north", "substation_south", "industrial_park"]

       for target_id in target_ids:
           PythonOperator(
               task_id=f"predict_{target_id}",
               python_callable=predict_target,
               op_kwargs={"target_id": target_id},
           )

.. mermaid:: /diagrams/user_guide/deployment_diagram_2.mmd

Cloud Deployment Options
------------------------

AWS
^^^

- **ECS Fargate** — run containerized predict/train tasks on a schedule via EventBridge rules
- **Lambda** — suitable for lightweight predictions with models under 250 MB (use Lambda layers or container images)
- **SageMaker** — use for training if GPU acceleration is needed; deploy OpenSTEF as a custom inference container

Azure
^^^^^

- **Container Instances** — on-demand containers triggered by Logic Apps or Azure Functions timer triggers
- **Azure Functions** — Python functions on a timer schedule; use Durable Functions for long-running training
- **Azure ML** — managed training pipelines with model registry integration

GCP
^^^

- **Cloud Run Jobs** — scheduled containerized workloads with automatic scaling to zero
- **Cloud Functions** — lightweight prediction triggers
- **Vertex AI** — managed training with custom containers

Model Storage in Production
----------------------------

OpenSTEF uses MLflow for model versioning and artifact storage. In production, point the tracking URI to a shared backend:

.. code-block:: python

   from openstef_models.integrations.mlflow import MLFlowStorage

   # Local filesystem (single-server deployments)
   storage = MLFlowStorage(
       tracking_uri="file:///models/mlflow",
       local_artifacts_path=Path("/models/artifacts"),
   )

   # Remote MLflow server (multi-service deployments)
   storage = MLFlowStorage(
       tracking_uri="http://mlflow-server:5000",
       local_artifacts_path=Path("/tmp/mlflow_cache"),
   )

For S3-backed artifact storage, configure MLflow's artifact root on the server side. OpenSTEF's ``MLFlowStorage`` integration handles model serialization and versioning transparently.

Monitoring and Alerting
-----------------------

Production forecasting systems need monitoring at three levels:

Pipeline health
^^^^^^^^^^^^^^^

Track whether jobs complete successfully and on time:

.. code-block:: python

   import time
   import logging

   logger = logging.getLogger(__name__)

   def monitored_predict(workflow, dataset, target_id: str):
       start = time.time()
       try:
           forecast = workflow.predict(dataset)
           duration = time.time() - start
           logger.info(
               "forecast_complete",
               extra={
                   "target_id": target_id,
                   "duration_seconds": duration,
                   "rows": len(forecast.data),
               },
           )
           # Emit metric to your monitoring system
           emit_metric("forecast.duration_seconds", duration, tags={"target": target_id})
           emit_metric("forecast.success", 1, tags={"target": target_id})
           return forecast
       except Exception as e:
           emit_metric("forecast.success", 0, tags={"target": target_id})
           logger.exception("Forecast failed for target %s", target_id)
           raise

Forecast quality
^^^^^^^^^^^^^^^^

Use OpenSTEF's evaluation framework to track model degradation over time. Schedule periodic backtests and compare metrics against baselines. The benchmarking pipeline (``openstef_beam.benchmarking``) automates this:

.. code-block:: python

   from openstef_beam.benchmarking import (
       BenchmarkPipeline,
       LocalBenchmarkStorage,
   )

   # Run periodic benchmark to detect model drift
   storage = LocalBenchmarkStorage(base_path=Path("/benchmarks"))

Data quality
^^^^^^^^^^^^

Monitor input data for missing values, stale timestamps, or out-of-range values before feeding it to the pipeline. Catch issues early to avoid silent forecast degradation.

Production Checklist
--------------------

Before going live, verify:

- **Model trained and validated** — run backtests covering representative periods
- **Data pipeline reliable** — input data arrives on time with alerting on gaps
- **Retry logic** — transient failures don't cause missed forecasts
- **Model versioning** — MLflow tracks which model version produced each forecast
- **Logging structured** — JSON logs with target IDs for easy filtering
- **Resource limits** — memory and CPU limits set on containers to prevent runaway jobs
- **Graceful degradation** — fallback to last-known-good forecast if prediction fails
- **Secrets management** — API keys and credentials stored in vault/secrets manager, not in code

Scaling to Many Targets
-----------------------

When forecasting hundreds of grid points or assets, parallelize across targets:

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor, as_completed
   from pathlib import Path

   def predict_single_target(target_id: str) -> dict:
       workflow = create_workflow(Path(f"/models/{target_id}"))
       dataset = load_data(target_id)
       forecast = workflow.predict(dataset)
       publish_forecast(target_id, forecast)
       return {"target_id": target_id, "status": "success"}

   target_ids = load_active_targets()  # e.g., from database

   with ProcessPoolExecutor(max_workers=8) as executor:
       futures = {executor.submit(predict_single_target, tid): tid for tid in target_ids}
       for future in as_completed(futures):
           result = future.result()
           logger.info("Completed: %s", result["target_id"])

For larger scale (1000+ targets), use distributed task queues (Celery) or Kubernetes Jobs with a job-per-target pattern.

.. mermaid:: /diagrams/user_guide/deployment_diagram_3.mmd
.. note:: [DIAGRAM: Scaling architecture showing job scheduler distributing targets across worker pool, with shared model storage and output sink]