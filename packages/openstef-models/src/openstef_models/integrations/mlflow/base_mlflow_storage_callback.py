# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Base configuration and shared utilities for MLflow storage callbacks.

Provides common fields, helper methods, and utility functions used by both
single-model and ensemble-model MLflow callbacks.
"""

import logging
from datetime import timedelta
from typing import Any, override

from mlflow.entities import Run
from pydantic import Field, PrivateAttr

from openstef_beam.evaluation import SubsetMetric
from openstef_beam.evaluation.metric_providers import MetricDirection
from openstef_core.base_model import BaseConfig
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.exceptions import (
    MissingColumnsError,
    ModelNotFoundError,
)
from openstef_core.types import Q, QuantileOrGlobal
from openstef_models.integrations.mlflow.mlflow_storage import MLFlowStorage
from openstef_models.models.base_forecasting_model import BaseForecastingModel


class BaseMLFlowStorageCallback(BaseConfig):
    """Base configuration and shared utilities for MLflow storage callbacks.

    Provides common fields and helper methods used by both single-model and
    ensemble-model MLflow callbacks. Not a callback itself — subclasses should
    also inherit from the appropriate callback type.
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

    def _find_run(self, model_id: str, run_name: str | None) -> Run | None:
        """Find an MLflow run by model_id and optional run_name.

        Args:
            model_id: The model identifier.
            run_name: Optional specific run name to search for.

        Returns:
            The MLflow Run object, or None if not found.
        """
        if run_name is not None:
            return self.storage.search_run(model_id=model_id, run_name=run_name)

        runs = self.storage.search_latest_runs(model_id=model_id)
        return next(iter(runs), None)

    def _try_load_model(
        self,
        run_id: str,
        model_id: str,
    ) -> BaseForecastingModel | None:
        """Try to load a model from MLflow, returning None on failure.

        Args:
            run_id: The MLflow run ID.
            model_id: The model identifier.

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

        if not isinstance(old_model, BaseForecastingModel):
            self._logger.warning(
                "Loaded old model from run %s is not a BaseForecastingModel, skipping model selection",
                run_id,
            )
            return None

        return old_model

    def _try_evaluate_model(
        self,
        run_id: str,
        old_model: BaseForecastingModel,
        input_data: TimeSeriesDataset,
    ) -> SubsetMetric | None:
        """Try to evaluate a model, returning None on failure.

        Args:
            run_id: The MLflow run ID (for logging).
            old_model: The model to evaluate.
            input_data: The dataset to evaluate on.

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
            True if the new model improves on the monitored metric.
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


__all__ = ["BaseMLFlowStorageCallback"]
