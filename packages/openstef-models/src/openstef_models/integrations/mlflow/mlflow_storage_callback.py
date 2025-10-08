# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""MLflow integration for tracking and storing forecasting workflows.

Provides storage and callback functionality to log model training runs, artifacts,
and metrics to MLflow. Automatically saves models, training data, and performance
metrics for each forecasting workflow execution.
"""

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast, override

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, Run
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig, BaseModel
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries.dataset import VersionedTimeSeriesDataset
from openstef_core.exceptions import ModelNotFoundError
from openstef_core.mixins import HyperParams, Stateful
from openstef_models.integrations.joblib import JoblibModelSerializer
from openstef_models.mixins import ModelIdentifier, ModelSerializer
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.workflows.forecasting_workflow import (
    ForecastingCallback,
    ForecastingWorkflow,
)

_RUN_ID_KEY = "mlflow.runId"


class MLFlowStorage(BaseConfig):
    """MLflow storage backend for managing training runs and model artifacts.

    Handles creation, storage, and retrieval of MLflow runs including models,
    training data, metrics, and hyperparameters. Organizes artifacts locally
    before uploading to MLflow tracking server.
    """

    tracking_uri: str = Field(default="./mlflow")
    local_artifacts_path: Path = Field(default=Path("./mlflow_artifacts_local"))
    experiment_name_prefix: str = Field(default="")
    # Artifact subdirectories
    data_path: str = Field(default="data")
    model_path: str = Field(default="model")

    model_serializer: ModelSerializer = Field(default_factory=JoblibModelSerializer)

    _client: MlflowClient = PrivateAttr()

    @override
    def model_post_init(self, context: Any) -> None:
        os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
        self._client = MlflowClient(tracking_uri=self.tracking_uri)

    def create_run(
        self,
        model_id: ModelIdentifier,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        experiment_tags: dict[str, str] | None = None,
        hyperparams: HyperParams | None = None,
    ) -> Run:
        """Create a new MLflow run for tracking a model training session.

        Creates or reuses an MLflow experiment named after the model ID, then
        starts a new run within that experiment. Logs hyperparameters if provided.

        Args:
            model_id: Unique identifier for the model, used as experiment name.
            run_name: Optional display name for this specific run.
            tags: Key-value pairs to attach to the run for filtering/organization.
            experiment_tags: Key-value pairs to attach to the experiment if created.
            hyperparams: Model hyperparameters to log with the run.

        Returns:
            Created MLflow Run object with run_id and metadata.
        """
        # Create experiment if not exists
        experiment = self._client.get_experiment_by_name(name=f"{self.experiment_name_prefix}{model_id}")
        if experiment is None:
            experiment_id = self._client.create_experiment(
                name=f"{self.experiment_name_prefix}{model_id}",
                tags=experiment_tags,
            )
        else:
            experiment_id = cast(str, experiment.experiment_id)

        # Create run
        run = self._client.create_run(
            experiment_id=experiment_id,
            tags=tags,
            run_name=run_name,
        )
        run_id: str = run.info.run_id

        # Log hyperparameters
        if hyperparams is not None:
            self._client.log_batch(
                run_id=run_id,
                params=[
                    Param(param_name, str(param_value)) for param_name, param_value in hyperparams.model_dump().items()
                ],
            )

        return run

    def finalize_run(
        self, model_id: ModelIdentifier, run_id: str, metrics: dict[str, float] | None = None, status: str = "FINISHED"
    ) -> None:
        """Complete an MLflow run by uploading artifacts and logging final metrics.

        Uploads all locally stored artifacts to MLflow, logs performance metrics,
        and marks the run as finished with the specified status.

        Args:
            model_id: Model identifier used to locate artifact path.
            run_id: MLflow run ID to finalize.
            metrics: Training/validation metrics to log (e.g., MAE, RMSE).
            status: Final run status, either "FINISHED", "FAILED", or "KILLED".
        """
        artifacts_path = self.get_artifacts_path(model_id=model_id, run_id=run_id)

        if artifacts_path.exists():
            self._client.log_artifacts(run_id=run_id, local_dir=str(artifacts_path.resolve()))

        # Log training metrics
        if metrics is not None:
            timestamp_now = datetime.now(tz=UTC).timestamp()
            self._client.log_batch(
                run_id=run_id,
                metrics=[
                    Metric(key=metric_name, value=metric_value, timestamp=timestamp_now * 1000, step=0)
                    for metric_name, metric_value in metrics.items()
                ],
            )

        # Mark the run as finished
        self._client.set_terminated(run_id=run_id, status=status)

    def search_latest_runs(
        self,
        model_id: ModelIdentifier,
        limit: int = 1,
        filter_string: str = "attribute.status = 'FINISHED'",
        order_by: Sequence[str] = ["start_time DESC"],
    ) -> list[Run]:
        """Search for recent runs of a specific model in MLflow.

        Queries MLflow for runs matching the filter criteria, ordered by most recent.
        Returns empty list if no experiment exists for the model.

        Args:
            model_id: Model identifier to search runs for.
            limit: Maximum number of runs to return.
            filter_string: MLflow filter query (e.g., status, metrics, tags).
            order_by: Sort order for results (e.g., ["start_time DESC"]).

        Returns:
            List of matching Run objects, newest first, up to limit count.
        """
        # Get related experiment
        experiment = self._client.get_experiment_by_name(name=f"{self.experiment_name_prefix}{model_id}")
        if experiment is None:
            return []

        # Find the latest successful run for this model
        return self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=list(order_by),
            max_results=limit,
        )

    def save_run_model(self, model_id: ModelIdentifier, run_id: str, model: Stateful) -> None:
        """Save a trained model to local artifacts directory for the run.

        Serializes the model using the configured serializer and stores it in
        the run's artifact directory. Model will be uploaded to MLflow when
        finalize_run is called.

        Args:
            model_id: Model identifier for organizing artifact paths.
            run_id: MLflow run ID to associate artifacts with.
            model: Trained model instance with state to serialize.
        """
        artifacts_path = self.get_artifacts_path(model_id=model_id, run_id=run_id)

        # Store the trained model
        model_path = artifacts_path / self.model_path
        model_path.mkdir(parents=True, exist_ok=True)
        with Path(model_path / f"model.{self.model_serializer.extension}").open("wb") as f:
            self.model_serializer.serialize(model, file=f)

    def load_run_model[T: Stateful](self, run_id: str, model: T) -> T:
        """Load a trained model from MLflow artifacts.

        Downloads model artifacts from MLflow and deserializes them into the
        provided model instance, restoring its trained state.

        Args:
            run_id: MLflow run ID containing the model artifacts.
            model: Model instance to load state into (must match saved type).

        Returns:
            Model instance with restored state from the run.
        """
        # Download and load the model
        with TemporaryDirectory() as tmpdir:
            self._client.download_artifacts(run_id=run_id, path=self.model_path, dst_path=tmpdir)
            with (Path(tmpdir) / self.model_path / f"model.{self.model_serializer.extension}").open("rb") as f:
                model = self.model_serializer.deserialize(model=model, file=f)

        return model

    def get_artifacts_path(self, model_id: ModelIdentifier, run_id: str | None = None) -> Path:
        """Get the local file system path for storing run artifacts.

        Constructs the directory path where artifacts are staged before uploading
        to MLflow. Path structure: local_artifacts_path/model_id/run_id.

        Args:
            model_id: Model identifier for organizing artifacts.
            run_id: Optional run ID to include in path. If None, returns model directory.

        Returns:
            Absolute path to the artifacts directory.
        """
        result = self.local_artifacts_path / str(model_id)
        if run_id is not None:
            result /= run_id

        return result


class MLFlowStorageCallback(BaseModel, ForecastingCallback):
    """MLFlow callback for logging forecasting workflow events."""

    storage: MLFlowStorage = Field(default_factory=MLFlowStorage)

    @override
    def on_fit_start(
        self,
        context: WorkflowContext[ForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ) -> None:
        run = self.storage.create_run(
            model_id=context.workflow.model_id,
            tags=context.workflow.model.tags,
            hyperparams=context.workflow.model.forecaster.hyperparams,
        )

        # Store run ID in context for later use
        context.data[_RUN_ID_KEY] = run.info.run_id

    @override
    def on_fit_end(
        self,
        context: WorkflowContext[ForecastingWorkflow],
        result: ModelFitResult,
    ) -> None:
        run_id: str | None = context.data.get(_RUN_ID_KEY)
        if run_id is None:
            raise RuntimeError("Run ID not set; on_fit_start must be called before on_fit_end")

        run_path = self.storage.get_artifacts_path(model_id=context.workflow.model_id, run_id=run_id)

        # Store the model input
        data_path = run_path / self.storage.data_path
        data_path.mkdir(parents=True, exist_ok=True)
        result.input_data_train.to_parquet(path=data_path / "data_train.parquet")
        if result.input_data_val is not None:
            result.input_data_val.to_parquet(path=data_path / "data_val.parquet")

        # Store the trained model
        self.storage.save_run_model(model_id=context.workflow.model_id, run_id=run_id, model=context.workflow.model)

        # Mark the run as finished
        self.storage.finalize_run(model_id=context.workflow.model_id, run_id=run_id, metrics=result.metrics)

    @override
    def on_predict_start(
        self, context: WorkflowContext[ForecastingWorkflow], data: VersionedTimeSeriesDataset | TimeSeriesDataset
    ):
        if context.workflow.model.is_fitted:
            return

        # Find the latest successful run for this model
        runs = self.storage.search_latest_runs(model_id=context.workflow.model_id)
        run = next(iter(runs), None)
        if run is None:
            raise ModelNotFoundError(model_id=context.workflow.model_id)

        # Load the model from the latest run
        run_id: str = run.info.run_id
        context.workflow.model = self.storage.load_run_model(run_id=run_id, model=context.workflow.model)
