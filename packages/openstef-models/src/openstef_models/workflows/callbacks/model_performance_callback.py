# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Model performance callback for forecasting workflows.

Evaluates model performance against a specified metric and threshold at the end of fitting.
If the model's performance does not meet the defined criteria, a ModelUnderperformingError is raised.
This allows for early stopping of workflows or using a fallback model when performance is insufficient.
"""

import logging
from typing import override

from pydantic import Field, PrivateAttr

from openstef_beam.evaluation.metric_providers import MetricDirection
from openstef_core.base_model import BaseConfig
from openstef_core.exceptions import ModelUnderperformingError
from openstef_core.types import Q, QuantileOrGlobal
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow, ForecastingCallback


class ModelPerformanceCallback(BaseConfig, ForecastingCallback):
    """Callback for comparing model performance against a treshold during the fit process.

    This callback evaluates the model's performance using a specified metric after fitting.
    If the performance metric does not meet the defined threshold, a ModelUnderperformingError is raised.
    """

    model_performance_metric_threshold: tuple[QuantileOrGlobal, str, MetricDirection, float] = Field(
        default=(Q(0.5), "R2", "higher_is_better", 0.0),
        description="Metric to monitor for model performance threshold during training.",
    )

    _logger: logging.Logger = PrivateAttr(default_factory=lambda: logging.getLogger(__name__))

    @override
    def on_fit_end(
        self,
        context: WorkflowContext[CustomForecastingWorkflow],
        result: ModelFitResult,
    ) -> None:
        """Evaluate model performance at the end of fitting and raise an error if underperforming.

        Args:
            context: The workflow context that completed fitting.
            result: Result of the fitting process containing performance metrics.

        Raises:
            ModelUnderperformingError: If the model's performance metric is below the defined threshold.
        """
        if result.metrics_val is None:
            self._logger.warning("No validation metrics found in fit results. Skipping performance evaluation.")
            return

        quantile, metric_name, metric_direction, threshold = self.model_performance_metric_threshold

        metric_value = result.metrics_val.get_metric(quantile, metric_name)

        if metric_value is None:
            self._logger.warning(
                "Performance metric '%s' not found in fit results. Skipping performance evaluation.",
                metric_name,
            )
            return

        match metric_direction:
            case "higher_is_better" if metric_value < threshold:
                raise ModelUnderperformingError(
                    metric_name=metric_name,
                    metric_value=metric_value,
                    threshold=threshold,
                )
            case "lower_is_better" if metric_value > threshold:
                raise ModelUnderperformingError(
                    metric_name=metric_name,
                    metric_value=metric_value,
                    threshold=threshold,
                )
            case _:
                return
