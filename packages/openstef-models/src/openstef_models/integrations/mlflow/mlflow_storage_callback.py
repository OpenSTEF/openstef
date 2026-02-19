# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""MLflow integration for tracking and storing forecasting workflows.

Provides a single callback for logging model training runs, artifacts,
and metrics to MLflow. Supports both single-model (ForecastingModel) and
ensemble (EnsembleForecastingModel) workflows via protocol-based dispatch.

Ensemble-specific behavior is enabled automatically when the model satisfies
the ``EnsembleModel`` and ``ExplainableEnsembleModel`` protocols, and when the
fit result satisfies ``EnsembleFitResult``:

- Logs combiner hyperparameters as the primary hyperparams
- Logs per-forecaster hyperparameters with name-prefixed keys
- Stores per-forecaster training data as separate artifacts
- Logs per-forecaster evaluation metrics with name-prefixed keys
- Stores feature importance plots for each explainable forecaster component
- Stores combiner feature importance plots
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast, override, runtime_checkable

from mlflow.entities import Run
from pydantic import Field, PrivateAttr

from openstef_beam.evaluation import SubsetMetric
from openstef_beam.evaluation.metric_providers import MetricDirection
from openstef_core.base_model import BaseConfig
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.datasets.versioned_timeseries_dataset import (
    VersionedTimeSeriesDataset,
)
from openstef_core.exceptions import (
    MissingColumnsError,
    ModelNotFoundError,
    SkipFitting,
)
from openstef_core.mixins import HyperParams
from openstef_core.types import Q, QuantileOrGlobal
from openstef_models.explainability import ExplainableForecaster
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting.forecaster import Forecaster
from openstef_models.models.forecasting_model import ForecastingModel, ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)


@runtime_checkable
class EnsembleModel(Protocol):
    """Protocol for ensemble models with multiple base forecasters."""

    @property
    def forecasters(self) -> dict[str, Forecaster]:
        """Return a dictionary of forecasters keyed by name."""
        ...


@runtime_checkable
class ExplainableEnsembleModel(Protocol):
    """Protocol for ensemble models with an explainable forecast combiner."""

    @property
    def combiner(self) -> ExplainableForecaster:
        """Return the explainable forecast combiner."""
        ...


@runtime_checkable
class EnsembleFitResult(Protocol):
    """Protocol for fit results that contain per-forecaster results."""

    @property
    def forecaster_fit_results(self) -> dict[str, ModelFitResult]:
        """Return per-forecaster fit results."""
        ...


class MLFlowStorageCallback(BaseConfig, ForecastingCallback):
    """MLFlow callback for logging forecasting workflow events.

    Handles both single-model and ensemble workflows via protocol-based
    dispatch.
    """

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
        context: WorkflowContext[CustomForecastingWorkflow],
        result: ModelFitResult,
    ) -> None:
        if self.model_selection_enable:
            self._run_model_selection(workflow=context.workflow, result=result)

        # Determine primary hyperparams based on model structure
        hyperparams = self._get_primary_hyperparams(context.workflow.model)

        # Create a new run
        run = self.storage.create_run(
            model_id=context.workflow.model_id,
            tags=context.workflow.model.tags,
            hyperparams=hyperparams,
            run_name=context.workflow.run_name,
            experiment_tags=context.workflow.experiment_tags,
        )
        run_id: str = run.info.run_id
        self._logger.info("Created MLflow run %s for model %s", run_id, context.workflow.model_id)

        # Log per-forecaster hyperparams for ensemble models
        if isinstance(context.workflow.model, EnsembleModel):
            self._log_forecaster_hyperparams(context.workflow.model, run_id)

        # Store the model input and per-forecaster data
        run_path = self.storage.get_artifacts_path(model_id=context.workflow.model_id, run_id=run_id)
        data_path = run_path / self.storage.data_path
        data_path.mkdir(parents=True, exist_ok=True)
        result.input_dataset.to_parquet(path=data_path / "data.parquet")
        self._logger.info("Stored training data at %s for run %s", data_path, run_id)

        if isinstance(result, EnsembleFitResult):
            self._store_forecaster_data(result.forecaster_fit_results, data_path)

        # Store feature importance plots
        if self.store_feature_importance_plot:
            self._store_feature_importances(context.workflow.model, data_path)

        # Store the trained model
        self.storage.save_run_model(
            model_id=context.workflow.model_id,
            run_id=run_id,
            model=context.workflow.model,
        )
        self._logger.info("Stored trained model for run %s", run_id)

        # Format the metrics for MLflow
        metrics = self._collect_metrics(result)

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

        run_id: str = run.info.run_id
        old_model = self.storage.load_run_model(run_id=run_id, model_id=context.workflow.model_id)

        if not isinstance(old_model, ForecastingModel):
            self._logger.warning(
                "Loaded model from run %s is not a ForecastingModel, cannot use for prediction",
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

    def _log_forecaster_hyperparams(self, model: EnsembleModel, run_id: str) -> None:
        """Log per-forecaster hyperparameters to the run."""
        for name, forecaster in model.forecasters.items():
            prefixed_params = {f"{name}.{k}": str(v) for k, v in forecaster.hyperparams.model_dump().items()}
            self.storage.log_hyperparams(run_id=run_id, params=prefixed_params)
            self._logger.debug("Logged hyperparams for forecaster '%s' in run %s", name, run_id)

    def _store_forecaster_data(self, forecaster_fit_results: dict[str, ModelFitResult], data_path: Path) -> None:
        """Store per-forecaster training data as separate parquet files."""
        for name, forecaster_result in forecaster_fit_results.items():
            forecaster_data_path = data_path / name
            forecaster_data_path.mkdir(parents=True, exist_ok=True)
            forecaster_result.input_dataset.to_parquet(path=forecaster_data_path / "data.parquet")
            self._logger.debug("Stored training data for forecaster '%s' at %s", name, forecaster_data_path)

    def _collect_metrics(self, result: ModelFitResult) -> dict[str, float]:
        """Collect all metrics from the fit result, including per-forecaster metrics for ensembles.

        Returns:
            Flat dictionary mapping metric names to values, including per-forecaster prefixed metrics.
        """
        metrics = self.metrics_to_dict(metrics=result.metrics_full, prefix="full_")
        metrics.update(self.metrics_to_dict(metrics=result.metrics_train, prefix="train_"))
        if result.metrics_val is not None:
            metrics.update(self.metrics_to_dict(metrics=result.metrics_val, prefix="val_"))
        if result.metrics_test is not None:
            metrics.update(self.metrics_to_dict(metrics=result.metrics_test, prefix="test_"))

        if isinstance(result, EnsembleFitResult):
            for name, forecaster_result in result.forecaster_fit_results.items():
                metrics.update(self.metrics_to_dict(metrics=forecaster_result.metrics_full, prefix=f"{name}_full_"))
                metrics.update(self.metrics_to_dict(metrics=forecaster_result.metrics_train, prefix=f"{name}_train_"))
                if forecaster_result.metrics_val is not None:
                    metrics.update(self.metrics_to_dict(metrics=forecaster_result.metrics_val, prefix=f"{name}_val_"))
                if forecaster_result.metrics_test is not None:
                    metrics.update(self.metrics_to_dict(metrics=forecaster_result.metrics_test, prefix=f"{name}_test_"))

        return metrics

    @staticmethod
    def _get_primary_hyperparams(model: ForecastingModel) -> HyperParams:
        """Determine primary hyperparameters from the model.

        For ensemble models: uses the combiner's hyperparameters.
        For single models: uses the forecaster's hyperparameters.

        Returns:
            The primary hyperparameters extracted from the model.
        """
        if isinstance(model, ExplainableEnsembleModel):
            config = getattr(model.combiner, "config", None)
            if config is not None:
                return getattr(config, "hyperparams", HyperParams())  # pyright: ignore[reportUnknownMemberType, reportReturnType]
        if model.forecaster is not None:
            return model.forecaster.hyperparams
        return HyperParams()

    @staticmethod
    def _store_feature_importances(model: ForecastingModel, data_path: Path) -> None:
        """Store feature importance plots for all explainable components of the model."""
        if isinstance(model, EnsembleModel):
            # Ensemble model: store per-forecaster feature importances
            for name, forecaster in model.forecasters.items():
                if isinstance(forecaster, ExplainableForecaster):
                    fig = forecaster.plot_feature_importances()
                    fig.write_html(data_path / f"feature_importances_{name}.html")  # pyright: ignore[reportUnknownMemberType]
        elif model.forecaster is not None and isinstance(model.forecaster, ExplainableForecaster):
            # Single model: store feature importance
            fig = model.forecaster.plot_feature_importances()
            fig.write_html(data_path / "feature_importances.html")  # pyright: ignore[reportUnknownMemberType]

        # Store combiner feature importances (if model has an explainable combiner)
        if isinstance(model, ExplainableEnsembleModel):
            combiner_fi = model.combiner.feature_importances
            if not combiner_fi.empty:
                fig = model.combiner.plot_feature_importances()
                fig.write_html(data_path / "feature_importances_combiner.html")  # pyright: ignore[reportUnknownMemberType]

    def _find_run(self, model_id: str, run_name: str | None) -> Run | None:
        """Find an MLflow run by model_id and optional run_name.

        Returns:
            The matching Run, or None if no run was found.
        """
        if run_name is not None:
            return self.storage.search_run(model_id=model_id, run_name=run_name)

        runs = self.storage.search_latest_runs(model_id=model_id)
        return next(iter(runs), None)

    def _try_load_model(self, run_id: str, model_id: str) -> ForecastingModel | None:
        """Try to load a model from MLflow, returning None on failure.

        Returns:
            The loaded model, or None if loading failed.
        """
        try:
            old_model = self.storage.load_run_model(run_id=run_id, model_id=model_id)
        except ModelNotFoundError:
            self._logger.warning(
                "Could not load model from previous run %s for model %s, skipping model selection",
                run_id,
                model_id,
            )
            return None

        if not isinstance(old_model, ForecastingModel):
            self._logger.warning(
                "Loaded old model from run %s is not a ForecastingModel, skipping model selection",
                run_id,
            )
            return None

        return old_model

    def _try_evaluate_model(
        self,
        run_id: str,
        old_model: ForecastingModel,
        input_data: TimeSeriesDataset,
    ) -> SubsetMetric | None:
        """Try to evaluate a model, returning None on failure.

        Returns:
            The evaluation metrics, or None if evaluation failed.
        """
        try:
            return old_model.score(input_data)
        except (MissingColumnsError, ValueError) as e:
            self._logger.warning(
                "Could not evaluate old model from run %s, skipping model selection: %s",
                run_id,
                e,
            )
            return None

    def _check_tags_compatible(self, run_tags: dict[str, str], new_tags: dict[str, str], run_id: str) -> bool:
        """Check if model tags are compatible, excluding mlflow.runName.

        Returns:
            True if tags are compatible, False otherwise.
        """
        old_tags = {k: v for k, v in run_tags.items() if k != "mlflow.runName"}

        if old_tags == new_tags:
            return True

        differences = {
            k: (old_tags.get(k), new_tags.get(k))
            for k in old_tags.keys() | new_tags.keys()
            if old_tags.get(k) != new_tags.get(k)
        }

        self._logger.info(
            "Model tags changed since run %s, skipping model selection. Changes: %s",
            run_id,
            differences,
        )
        return False

    def _check_is_new_model_better(
        self,
        old_metrics: SubsetMetric,
        new_metrics: SubsetMetric,
    ) -> bool:
        """Compare old and new model metrics to determine if the new model is better.

        Returns:
            True if the new model is better, False otherwise.
        """
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

    @staticmethod
    def metrics_to_dict(metrics: SubsetMetric, prefix: str) -> dict[str, float]:
        """Convert SubsetMetric to a flat dictionary for MLflow logging.

        Args:
            metrics: The metrics to convert.
            prefix: Prefix to add to each metric key (e.g. "full_", "train_").

        Returns:
            Flat dictionary mapping metric names to values.
        """
        return {
            f"{prefix}{quantile}_{metric_name}": value
            for quantile, metrics_dict in metrics.metrics.items()
            for metric_name, value in metrics_dict.items()
        }


__all__ = [
    "EnsembleFitResult",
    "EnsembleModel",
    "ExplainableEnsembleModel",
    "MLFlowStorageCallback",
]
