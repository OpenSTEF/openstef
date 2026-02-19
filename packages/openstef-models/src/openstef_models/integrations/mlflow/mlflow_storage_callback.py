# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""MLflow integration for tracking and storing forecasting workflows.

Provides callback functionality to log model training runs, artifacts,
and metrics to MLflow. Automatically saves models, training data, and performance
metrics for each forecasting workflow execution.
"""

from datetime import UTC, datetime
from typing import cast, override

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset import (
    VersionedTimeSeriesDataset,
)
from openstef_core.exceptions import (
    ModelNotFoundError,
    SkipFitting,
)
from openstef_models.explainability import ExplainableForecaster
from openstef_models.integrations.mlflow.base_mlflow_storage_callback import (
    BaseMLFlowStorageCallback,
)
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.base_forecasting_model import BaseForecastingModel
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)


class MLFlowStorageCallback(BaseMLFlowStorageCallback, ForecastingCallback):
    """MLFlow callback for logging forecasting workflow events."""

    @override
    def on_fit_start(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ) -> None:
        if not self.model_reuse_enable:
            return

        run = self._find_run(model_id=context.workflow.model_id, run_name=context.workflow.run_name)

        if run is not None:
            # Check if the run is recent enough to skip re-fitting
            now = datetime.now(tz=UTC)
            end_time_millis = cast(float | None, run.info.end_time)
            run_end_datetime = (
                datetime.fromtimestamp(end_time_millis / 1000, tz=UTC) if end_time_millis is not None else None
            )
            self._logger.info(
                "Found previous MLflow run %s for model %s ended at %s",
                cast(str, run.info.run_id),
                context.workflow.model_id,
                run_end_datetime,
            )
            if run_end_datetime is not None and (now - run_end_datetime) <= self.model_reuse_max_age:
                raise SkipFitting("Model is recent enough, skipping re-fit.")

    @override
    def on_fit_end(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        result: ModelFitResult,
    ) -> None:
        if self.model_selection_enable:
            self._run_model_selection(workflow=context.workflow, result=result)

        # Create a new run
        model = context.workflow.model
        run = self.storage.create_run(
            model_id=context.workflow.model_id,
            tags=model.tags,
            hyperparams=context.workflow.model.forecaster.hyperparams,
            run_name=context.workflow.run_name,
            experiment_tags=context.workflow.experiment_tags,
        )
        run_id: str = run.info.run_id
        self._logger.info("Created MLflow run %s for model %s", run_id, context.workflow.model_id)

        # Store the model input
        run_path = self.storage.get_artifacts_path(model_id=context.workflow.model_id, run_id=run_id)
        data_path = run_path / self.storage.data_path
        data_path.mkdir(parents=True, exist_ok=True)
        result.input_dataset.to_parquet(path=data_path / "data.parquet")
        self._logger.info("Stored training data at %s for run %s", data_path, run_id)

        # Store feature importance plots if enabled
        if self.store_feature_importance_plot and isinstance(model.forecaster, ExplainableForecaster):
            fig = model.forecaster.plot_feature_importances()
            fig.write_html(data_path / "feature_importances.html")  # pyright: ignore[reportUnknownMemberType]

        # Store the trained model
        self.storage.save_run_model(
            model_id=context.workflow.model_id,
            run_id=run_id,
            model=context.workflow.model,
        )
        self._logger.info("Stored trained model for run %s", run_id)

        # Format the metrics for MLflow
        metrics = self.metrics_to_dict(metrics=result.metrics_full, prefix="full_")
        metrics.update(self.metrics_to_dict(metrics=result.metrics_train, prefix="train_"))
        if result.metrics_val is not None:
            metrics.update(self.metrics_to_dict(metrics=result.metrics_val, prefix="val_"))
        if result.metrics_test is not None:
            metrics.update(self.metrics_to_dict(metrics=result.metrics_test, prefix="test_"))

        # Mark the run as finished
        self.storage.finalize_run(model_id=context.workflow.model_id, run_id=run_id, metrics=metrics)
        self._logger.info("Stored MLflow run %s for model %s", run_id, context.workflow.model_id)

    @override
    def on_predict_start(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ):
        if context.workflow.model.is_fitted:
            return

        run = self._find_run(model_id=context.workflow.model_id, run_name=context.workflow.run_name)

        if run is None:
            raise ModelNotFoundError(model_id=context.workflow.model_id)

        # Load the model from the run
        run_id: str = run.info.run_id

        old_model = self.storage.load_run_model(run_id=run_id, model_id=context.workflow.model_id)

        if not isinstance(old_model, BaseForecastingModel):
            self._logger.warning(
                "Loaded model from run %s is not a BaseForecastingModel, cannot use for prediction",
                cast(str, run.info.run_id),
            )
            return

        context.workflow.model = old_model  # pyright: ignore[reportAttributeAccessIssue]
        self._logger.info(
            "Loaded model from MLflow run %s for model %s",
            run_id,
            context.workflow.model_id,
        )

    def _run_model_selection(self, workflow: CustomForecastingWorkflow, result: ModelFitResult) -> None:
        run = self._find_run(model_id=workflow.model_id, run_name=None)
        if run is None:
            return

        run_id = cast(str, run.info.run_id)

        if not self._check_tags_compatible(
            run_tags=run.data.tags,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            new_tags=workflow.model.tags,
            run_id=run_id,
        ):
            return

        new_model = workflow.model
        new_metrics = result.metrics_full

        old_model = self._try_load_model(run_id=run_id, model_id=workflow.model_id)

        if old_model is None:
            return

        old_metrics = self._try_evaluate_model(
            run_id=run_id,
            old_model=old_model,
            input_data=result.input_dataset,
        )

        if old_metrics is None:
            return

        if self._check_is_new_model_better(old_metrics=old_metrics, new_metrics=new_metrics):
            workflow.model = new_model  # pyright: ignore[reportAttributeAccessIssue]
        else:
            workflow.model = old_model  # pyright: ignore[reportAttributeAccessIssue]
            self._logger.info(
                "New model did not improve %s metric from previous run %s, reusing old model",
                self.model_selection_metric,
                run_id,
            )
            raise SkipFitting("New model did not improve monitored metric, skipping re-fit.")


__all__ = ["MLFlowStorageCallback"]
