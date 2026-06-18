# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Backtesting integration with openstef-beam.

Bridges beam's backtesting interface
(:class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin`)
to an OpenSTEF
:class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow`.
:class:`FoundationModelBacktestForecaster` wraps a **single, already-built** workflow
instance and reuses it across every backtest window, so an expensive backend (e.g. a
loaded ONNX session) is created once and shared — there is no per-window model loading.
Every window is forecast by calling the workflow's own
:meth:`~CustomForecastingWorkflow.predict`, so the model's preprocessing (feature
selection / covariates) and postprocessing (quantile sorting) apply uniformly.

Forecasting runs one window at a time. Batching multiple windows into a single backend
call is a separate, planned optimisation and is intentionally not done here.

Foundation models such as Chronos-2 are zero-shot, so :meth:`fit` is a no-op and the
default window config disables training.
"""

from datetime import timedelta
from typing import override

from pydantic import Field

from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestForecasterConfig,
    BacktestForecasterMixin,
)
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import Quantile
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

#: Default backtest window settings for a zero-shot, load-once foundation model: training
#: disabled, a generous 60-day context (the model truncates it to its own window
#: internally), and a 48-hour prediction length matching the default forecasting horizon.
#: Tweak one field with ``DEFAULT_BACKTEST_CONFIG.model_copy(update={...})`` to match a
#: model configured for a different horizon.
DEFAULT_BACKTEST_CONFIG = BacktestForecasterConfig(
    requires_training=False,
    predict_length=timedelta(hours=48),
    predict_min_length=timedelta(minutes=15),
    predict_context_length=timedelta(days=60),
    predict_context_min_coverage=0.0,
    training_context_length=timedelta(0),
    training_context_min_coverage=0.0,
)


class FoundationModelBacktestForecaster(BaseModel, BacktestForecasterMixin):
    """Backtest wrapper around a single, shared forecasting workflow.

    The wrapped :attr:`workflow` is built once and reused for every prediction window
    (load-once). Each window is forecast through
    :meth:`~CustomForecastingWorkflow.predict` with the window horizon as the forecast
    start, so the adapter never reaches into the workflow's model.

    Construct it with just a workflow — :attr:`config` then defaults to a zero-shot,
    load-once setup. Pass an explicit :attr:`config` to override the window settings::

        adapter = FoundationModelBacktestForecaster(workflow=workflow)
    """

    workflow: CustomForecastingWorkflow = Field(
        description="The shared, pre-built forecasting workflow to run for every window."
    )
    config: BacktestForecasterConfig = Field(
        default=DEFAULT_BACKTEST_CONFIG,
        description="Backtest window configuration. Defaults to a zero-shot, load-once setup with a 48-hour "
        "prediction length; pass an explicit config to match a model configured for a different horizon.",
    )

    @property
    @override
    def quantiles(self) -> list[Quantile]:
        return self.workflow.model.quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """No-op: foundation models are zero-shot and need no per-window training."""

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Forecast a single backtest window through the workflow.

        Returns:
            The workflow forecast for the window, or ``None`` when there is no
            observed target history before the horizon (no reliable forecast can
            be produced).
        """
        window = data.get_window(
            start=data.horizon - self.config.predict_context_length,
            end=data.horizon + self.config.predict_length,
            available_before=data.horizon,
        )

        target = window.data[self.workflow.model.target_column]
        if target[target.index < data.horizon].notna().sum() == 0:
            return None

        return self.workflow.predict(data=window, forecast_start=data.horizon)


__all__ = [
    "DEFAULT_BACKTEST_CONFIG",
    "FoundationModelBacktestForecaster",
]
