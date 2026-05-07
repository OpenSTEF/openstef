# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Model performance callback for forecasting workflows.

Evaluates model performance against a specified metric and threshold at the end of fitting.
If the model's performance does not meet the defined criteria, a ModelUnderperfomingError is raised.
This allows for early stopping of workflows or using a fallback model when performance is insufficient.
"""

import logging
from typing import override

from pydantic import Field, PrivateAttr

from openstef_beam.evaluation.metric_providers import MetricDirection
from openstef_core.base_model import BaseConfig
from openstef_core.exceptions import ModelUnderperfomingError
from openstef_core.types import QuantileOrGlobal
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow, ForecastingCallback


class ModelPerformanceCallback(BaseConfig, ForecastingCallback):
    """Callback for comparing model performance against a treshold during the fit process.

    This callback evaluates the model's performance using a specified metric after fitting.
    If the performance metric does not meet the defined threshold, a ModelUnderperfomingError is raised.
    """

    metric_name: str = Field(description="The name of the performance metric to evaluate.")
    threshold: float = Field(
        description="The minimum acceptable value for the performance metric. If the model's performance is "
        "below or above this threshold (depending on `metric_direction`), it will be considered underperforming."
    )
    metric_direction: MetricDirection = Field(
        description=("Direction of the performance metric. Either 'higher_is_better' or 'lower_is_better'.")
    )
    quantile: QuantileOrGlobal = Field(
        default="global",
        description=(
            "The quantile level to evaluate the metric on. Use 'global' for overall performance metrics, or specify a "
            "quantile (e.g., 0.5 for median) for quantile-specific metrics."
        ),
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
            ModelUnderperfomingError: If the model's performance metric is below the defined threshold.
        """
        if result.metrics_val is None:
            self._logger.warning("No validation metrics found in fit results. Skipping performance evaluation.")
            return

        metric_value = result.metrics_val.get_metric(self.quantile, self.metric_name)

        if metric_value is None:
            self._logger.warning(
                "Performance metric '%s' not found in fit results. Skipping performance evaluation.",
                self.metric_name,
            )
            return

        match self.metric_direction:
            case "higher_is_better" if metric_value < self.threshold:
                raise ModelUnderperfomingError(
                    metric_name=self.metric_name,
                    metric_value=metric_value,
                    threshold=self.threshold,
                )
            case "lower_is_better" if metric_value > self.threshold:
                raise ModelUnderperfomingError(
                    metric_name=self.metric_name,
                    metric_value=metric_value,
                    threshold=self.threshold,
                )
            case _:
                return
