# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

from typing import Any, cast

import pytest

from openstef_beam.evaluation.metric_providers import MetricDirection
from openstef_core.exceptions import ModelUnderperformingError
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.workflows.callbacks.model_performance_callback import ModelPerformanceCallback
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow


def _make_callback(
    metric_name: str = "MAE",
    threshold: float = 10.0,
    metric_direction: MetricDirection = "lower_is_better",
) -> ModelPerformanceCallback:
    return ModelPerformanceCallback(
        metric_name=metric_name,
        threshold=threshold,
        metric_direction=metric_direction,
        quantile="global",
    )


def _make_result(metric_value: float | None) -> MagicMock:
    result = MagicMock()
    result.metrics_val = MagicMock()
    result.metrics_val.get_metric.return_value = metric_value
    return result


def _make_context() -> WorkflowContext[CustomForecastingWorkflow]:
    return cast(WorkflowContext[CustomForecastingWorkflow], WorkflowContext(workflow=MagicMock()))


def test_on_fit_end__no_metrics_val__skips_without_error() -> None:
    callback = _make_callback()
    context = _make_context()
    result = MagicMock()
    result.metrics_val = None

    callback.on_fit_end(context=context, result=result)


def test_on_fit_end__metric_not_found__skips_without_error() -> None:
    callback = _make_callback()
    context = _make_context()
    result = _make_result(metric_value=None)

    callback.on_fit_end(context=context, result=result)


def test_on_fit_end__lower_is_better__metric_above_threshold__raises() -> None:
    callback = _make_callback(metric_name="MAE", threshold=10.0, metric_direction="lower_is_better")
    context = _make_context()
    result = _make_result(metric_value=15.0)

    with pytest.raises(ModelUnderperformingError):
        callback.on_fit_end(context=context, result=result)


def test_on_fit_end__lower_is_better__metric_below_threshold__no_error() -> None:
    callback = _make_callback(metric_name="MAE", threshold=10.0, metric_direction="lower_is_better")
    context = _make_context()
    result = _make_result(metric_value=5.0)

    callback.on_fit_end(context=context, result=result)


def test_on_fit_end__higher_is_better__metric_below_threshold__raises() -> None:
    callback = _make_callback(metric_name="R2", threshold=0.8, metric_direction="higher_is_better")
    context = _make_context()
    result = _make_result(metric_value=0.5)

    with pytest.raises(ModelUnderperformingError):
        callback.on_fit_end(context=context, result=result)


def test_on_fit_end__higher_is_better__metric_above_threshold__no_error() -> None:
    callback = _make_callback(metric_name="R2", threshold=0.8, metric_direction="higher_is_better")
    context = _make_context()
    result = _make_result(metric_value=0.95)

    callback.on_fit_end(context=context, result=result)
