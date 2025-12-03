# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""MLflow integration for tracking and storing forecasting workflows.

Provides storage and callback functionality to log model training runs, artifacts,
and metrics to MLflow. Automatically saves models, training data, and performance
metrics for each forecasting workflow execution.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast, override

from pydantic import Field, PrivateAttr

from openstef_beam.evaluation import SubsetMetric
from openstef_beam.evaluation.metric_providers import MetricDirection
from openstef_core.base_model import BaseConfig
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.exceptions import ModelNotFoundError, SkipFitting
from openstef_core.types import Q, QuantileOrGlobal
from openstef_models.explainability import ExplainableForecaster
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models import ForecastingModel
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)


class MLFlowStorageCallback(BaseConfig, ForecastingCallback):
    """MLFlow callback for logging forecasting workflow events."""

    storage: MLFlowStorage = Field(default_factory=MLFlowStorage)

    model_reuse_enable: bool = Field(default=True)
    model_reuse_max_age: timedelta = Field(default=timedelta(days=7))

    model_selection_enable: bool = Field(default=True)
    model_selection_metric: tuple[QuantileOrGlobal, str, MetricDirection] = Field(
        default=(Q(0.5), "R2", "higher_is_better"),
        description="Metric to monitor for model performance when retraining.",
    )
    model_selection_old_model_penalty: float = Field(
        default=1.2,
        description="Penalty to apply to the old model's metric to bias selection towards newer models.",
    )

    store_feature_importance_plot: bool = Field(
        default=True,
        description="Whether to store feature importance plots in MLflow artifacts if available.",
    )

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @override
    def model_post_init(self, context: Any) -> None:
        pass

    @override
    def on_fit_start(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ) -> None:
        if not self.model_reuse_enable:
            return

        # Find the latest successful run for this model
        runs = self.storage.search_latest_runs(model_id=context.workflow.model_id)
        run = next(iter(runs), None)

        if run is not None:
            # Check if the run is recent enough to skip re-fitting
            now = datetime.now(tz=UTC)
            run_end_datetime = datetime.fromtimestamp(cast(float, run.info.end_time) / 1000, tz=UTC)
            self._logger.info(
                "Found previous MLflow run %s for model %s ended at %s",
                cast(str, run.info.run_id),
                context.workflow.model_id,
                run_end_datetime,
            )
            if (now - run_end_datetime) <= self.model_reuse_max_age:
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
        run = self.storage.create_run(
            model_id=context.workflow.model_id,
            tags=context.workflow.model.tags,
            hyperparams=context.workflow.model.forecaster.hyperparams,  # type: ignore TODO Make MLFlow compatible with OpenSTEF Meta
        )
        run_id: str = run.info.run_id
        self._logger.info("Created MLflow run %s for model %s", run_id, context.workflow.model_id)

        # Store the model input
        run_path = self.storage.get_artifacts_path(model_id=context.workflow.model_id, run_id=run_id)
        data_path = run_path / self.storage.data_path
        data_path.mkdir(parents=True, exist_ok=True)
        result.input_dataset.to_parquet(path=data_path / "data.parquet")
        self._logger.info("Stored training data at %s for run %s", data_path, run_id)

        # Store feature importance plot if enabled
        if (
            self.store_feature_importance_plot
            and isinstance(context.workflow.model, ForecastingModel)
            and isinstance(context.workflow.model.forecaster, ExplainableForecaster)
        ):
            fig = context.workflow.model.forecaster.plot_feature_importances()
            fig.write_html(data_path / "feature_importances.html")  # pyright: ignore[reportUnknownMemberType]

        # Store the trained model
        self.storage.save_run_model(model_id=context.workflow.model_id, run_id=run_id, model=context.workflow.model)
        self._logger.info("Stored trained model for run %s", run_id)

        # Format the metrics for MLflow
        metrics = _metrics_to_dict(metrics=result.metrics_full, prefix="full_")
        metrics.update(_metrics_to_dict(metrics=result.metrics_train, prefix="train_"))
        if result.metrics_val is not None:
            metrics.update(_metrics_to_dict(metrics=result.metrics_val, prefix="val_"))
        if result.metrics_test is not None:
            metrics.update(_metrics_to_dict(metrics=result.metrics_test, prefix="test_"))

        # Mark the run as finished
        self.storage.finalize_run(model_id=context.workflow.model_id, run_id=run_id, metrics=metrics)
        self._logger.info("Stored MLflow run %s for model %s", run_id, context.workflow.model_id)

    @override
    def on_predict_start(
        self, context: WorkflowContext[CustomForecastingWorkflow], data: VersionedTimeSeriesDataset | TimeSeriesDataset
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
        old_model = self.storage.load_run_model(run_id=run_id)
        if not isinstance(old_model, ForecastingModel):
            self._logger.warning(
                "Loaded model from run %s is not a ForecastingModel, cannot use for prediction",
                cast(str, run.info.run_id),
            )
            return

        context.workflow.model = old_model
        self._logger.info("Loaded model from MLflow run %s for model %s", run_id, context.workflow.model_id)

    def _run_model_selection(self, workflow: CustomForecastingWorkflow, result: ModelFitResult) -> None:
        # Find the latest successful run for this model
        runs = self.storage.search_latest_runs(model_id=workflow.model_id)
        run = next(iter(runs), None)
        if run is None:
            return

        # Backup the new model
        new_model = workflow.model
        new_metrics = result.metrics_full

        # Restore the old model and evaluate
        old_model = self.storage.load_run_model(run_id=cast(str, run.info.run_id))
        if not isinstance(old_model, ForecastingModel):
            self._logger.warning(
                "Loaded old model from run %s is not a ForecastingModel, skipping model selection",
                cast(str, run.info.run_id),
            )
            return
        old_metrics = old_model.score(result.input_dataset)

        if self._check_is_new_model_better(old_metrics=old_metrics, new_metrics=new_metrics):
            workflow.model = new_model
        else:
            workflow.model = old_model
            self._logger.info(
                "New model did not improve %s metric from previous run %s, reusing old model",
                self.model_selection_metric,
                cast(str, run.info.run_id),
            )
            raise SkipFitting("New model did not improve monitored metric, skipping re-fit.")

    def _check_is_new_model_better(
        self,
        old_metrics: SubsetMetric,
        new_metrics: SubsetMetric,
    ) -> bool:
        quantile, metric_name, direction = self.model_selection_metric

        old_metric = old_metrics.get_metric(quantile=quantile, metric_name=metric_name)
        new_metric = new_metrics.get_metric(quantile=quantile, metric_name=metric_name)

        if old_metric is None or new_metric is None:
            self._logger.warning(
                "Could not find %s metric for quantile %s in old or new model metrics, assuming improvement",
                metric_name,
                quantile,
            )
            return True

        self._logger.info(
            "Comparing old model %s metric %.5f to new model %s metric %.5f for quantile %s",
            metric_name,
            old_metric,
            metric_name,
            new_metric,
            quantile,
        )

        match direction:
            case "higher_is_better" if new_metric >= old_metric / self.model_selection_old_model_penalty:
                return True
            case "lower_is_better" if new_metric <= old_metric / self.model_selection_old_model_penalty:
                return True
            case _:
                return False


def _metrics_to_dict(metrics: SubsetMetric, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}{quantile}_{metric_name}": value
        for quantile, metrics_dict in metrics.metrics.items()
        for metric_name, value in metrics_dict.items()
    }


__all__ = ["MLFlowStorageCallback"]
