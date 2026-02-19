# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""MLflow storage callback for ensemble forecasting models.

Provides MLflow storage and tracking for ensemble forecasting workflows using
composition with MLFlowStorageCallbackBase rather than inheriting from
MLFlowStorageCallback. This avoids conflicting generic type parameters and
keeps the callback fully type-safe.

Ensemble-specific behavior:
- Logs combiner hyperparameters as the primary hyperparams
- Logs per-forecaster hyperparameters with name-prefixed keys
- Stores feature importance plots for each explainable forecaster component
"""

import logging
from datetime import UTC, datetime
from typing import cast, override

from pydantic import PrivateAttr

from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset import VersionedTimeSeriesDataset
from openstef_core.exceptions import ModelNotFoundError, SkipFitting
from openstef_meta.models.ensemble_forecasting_model import EnsembleModelFitResult
from openstef_meta.workflows import CustomEnsembleForecastingWorkflow, EnsembleForecastingCallback
from openstef_models.explainability import ExplainableForecaster
from openstef_models.integrations.mlflow.mlflow_storage_callback import BaseMLFlowStorageCallback, metrics_to_dict
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.base_forecasting_model import BaseForecastingModel


class EnsembleMLFlowStorageCallback(BaseMLFlowStorageCallback, EnsembleForecastingCallback):
    """MLFlow callback for ensemble forecasting workflows.

    Uses composition with MLFlowStorageCallbackBase for shared MLflow storage
    configuration and utility methods, combined with EnsembleForecastingCallback
    for properly-typed ensemble workflow hooks.

    Handles EnsembleForecastingModel instances by:
    - Logging combiner hyperparameters as the primary model hyperparams
    - Logging per-forecaster hyperparameters with name-prefixed keys
    - Storing feature importance plots for each explainable base forecaster
    """

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @override
    def on_fit_start(
        self,
        context: WorkflowContext[CustomEnsembleForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ) -> None:
        """Check model reuse before fitting.

        Raises:
            SkipFitting: If a recent model already exists.
        """
        if not self.model_reuse_enable:
            return

        run = self._find_run(model_id=context.workflow.model_id, run_name=context.workflow.run_name)

        if run is not None:
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
        context: WorkflowContext[CustomEnsembleForecastingWorkflow],
        result: EnsembleModelFitResult,
    ) -> None:
        """Store ensemble model, hyperparams, artifacts, and metrics to MLflow."""
        if self.model_selection_enable:
            self._run_model_selection(workflow=context.workflow, result=result)

        model = context.workflow.model

        # Create a new run with combiner hyperparameters
        run = self.storage.create_run(
            model_id=context.workflow.model_id,
            tags=model.tags,
            hyperparams=model.combiner.config.hyperparams,
            run_name=context.workflow.run_name,
            experiment_tags=context.workflow.experiment_tags,
        )
        run_id: str = run.info.run_id
        self._logger.info("Created MLflow run %s for model %s", run_id, context.workflow.model_id)

        # Log per-forecaster hyperparameters
        for name, forecaster in model.forecasters.items():
            hyperparams = forecaster.hyperparams
            prefixed_params = {f"{name}.{k}": str(v) for k, v in hyperparams.model_dump().items()}
            self.storage.log_hyperparams(run_id=run_id, params=prefixed_params)
            self._logger.debug("Logged hyperparams for forecaster '%s' in run %s", name, run_id)

        # Store the model input
        run_path = self.storage.get_artifacts_path(model_id=context.workflow.model_id, run_id=run_id)
        data_path = run_path / self.storage.data_path
        data_path.mkdir(parents=True, exist_ok=True)
        result.input_dataset.to_parquet(path=data_path / "data.parquet")
        self._logger.info("Stored training data at %s for run %s", data_path, run_id)

        # Store feature importance plots for each explainable forecaster
        if self.store_feature_importance_plot:
            for name, forecaster in model.forecasters.items():
                if isinstance(forecaster, ExplainableForecaster):
                    fig = forecaster.plot_feature_importances()
                    fig.write_html(data_path / f"feature_importances_{name}.html")  # pyright: ignore[reportUnknownMemberType]

        # Store the trained model
        self.storage.save_run_model(
            model_id=context.workflow.model_id,
            run_id=run_id,
            model=context.workflow.model,
        )
        self._logger.info("Stored trained model for run %s", run_id)

        # Format the metrics for MLflow
        metrics = metrics_to_dict(metrics=result.metrics_full, prefix="full_")
        metrics.update(metrics_to_dict(metrics=result.metrics_train, prefix="train_"))
        if result.metrics_val is not None:
            metrics.update(metrics_to_dict(metrics=result.metrics_val, prefix="val_"))
        if result.metrics_test is not None:
            metrics.update(metrics_to_dict(metrics=result.metrics_test, prefix="test_"))

        # Mark the run as finished
        self.storage.finalize_run(model_id=context.workflow.model_id, run_id=run_id, metrics=metrics)
        self._logger.info("Stored MLflow run %s for model %s", run_id, context.workflow.model_id)

    @override
    def on_predict_start(
        self,
        context: WorkflowContext[CustomEnsembleForecastingWorkflow],
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ):
        """Load ensemble model from MLflow for prediction.

        Raises:
            ModelNotFoundError: If no model run is found.
        """
        if context.workflow.model.is_fitted:
            return

        run = self._find_run(model_id=context.workflow.model_id, run_name=context.workflow.run_name)

        if run is None:
            raise ModelNotFoundError(model_id=context.workflow.model_id)

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

    def _run_model_selection(self, workflow: CustomEnsembleForecastingWorkflow, result: EnsembleModelFitResult) -> None:
        """Compare new ensemble model against the previous best and keep the better one.

        Raises:
            SkipFitting: If the new model does not improve on the monitored metric.
        """
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


__all__ = ["EnsembleMLFlowStorageCallback"]
