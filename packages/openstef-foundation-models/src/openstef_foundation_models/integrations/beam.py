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

Foundation models such as Chronos-2 are zero-shot, so :meth:`fit` is a no-op and the
default window config disables training.

Forecasting can run either one window at a time (:meth:`predict`) or with multiple
windows stacked into a single backend call (:meth:`predict_batch`, enabled by setting
:attr:`batch_size`), which beam's :class:`BacktestPipeline` selects automatically.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Self, override

from pydantic import Field

from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestBatchForecasterMixin,
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
#: internally), and a 48-hour prediction length. :meth:`FoundationModelBacktestForecaster.from_workflow`
#: tailors the prediction length to a specific workflow's horizon.
DEFAULT_BACKTEST_CONFIG = BacktestForecasterConfig(
    requires_training=False,
    predict_length=timedelta(hours=48),
    predict_min_length=timedelta(minutes=15),
    predict_context_length=timedelta(days=60),
    predict_context_min_coverage=0.0,
    training_context_length=timedelta(0),
    training_context_min_coverage=0.0,
)


class FoundationModelBacktestForecaster(BaseModel, BacktestBatchForecasterMixin, BacktestForecasterMixin):
    """Backtest wrapper around a single, shared forecasting workflow.

    The wrapped :attr:`workflow` is built once and reused for every prediction window
    (load-once). Each window is forecast through
    :meth:`~CustomForecastingWorkflow.predict` with the window horizon as the forecast
    start, so the adapter never reaches into the workflow's model.

    Setting :attr:`batch_size` greater than 1 makes beam's pipeline route consecutive
    prediction windows through :meth:`predict_batch`, which issues a single batched
    ``workflow.predict_batch`` call (one backend call inside the forecaster) instead of
    one call per window.

    Prefer :meth:`from_workflow`, which sizes the window to the workflow's own horizon::

        adapter = FoundationModelBacktestForecaster.from_workflow(workflow)

    The raw constructor takes a workflow and a window :attr:`config`; the config defaults
    to a generic zero-shot, load-once setup (:data:`DEFAULT_BACKTEST_CONFIG`).
    """

    workflow: CustomForecastingWorkflow = Field(
        description="The shared, pre-built forecasting workflow to run for every window."
    )
    config: BacktestForecasterConfig = Field(
        default=DEFAULT_BACKTEST_CONFIG,
        description="Backtest window configuration. Defaults to a generic zero-shot, load-once setup; "
        "use from_workflow to size the prediction length to the workflow's horizon.",
    )
    batch_size: int | None = Field(
        default=None,
        description="Max prediction windows stacked into one backend call. None or 1 means one window at a time.",
    )

    @classmethod
    def from_workflow(
        cls,
        workflow: CustomForecastingWorkflow,
        *,
        predict_length: timedelta | None = None,
        predict_context_length: timedelta = DEFAULT_BACKTEST_CONFIG.predict_context_length,
        batch_size: int | None = None,
    ) -> Self:
        """Build a load-once adapter with a zero-shot window config sized to *workflow*.

        Args:
            workflow: The pre-built workflow to reuse across every backtest window.
            predict_length: Forecast horizon per window. Defaults to the workflow's
                maximum configured horizon.
            predict_context_length: History fed to the model as context.
            batch_size: Max prediction windows stacked into one backend call. ``None``
                or ``1`` forecasts one window at a time.

        Returns:
            A configured adapter wrapping *workflow*.
        """
        config = DEFAULT_BACKTEST_CONFIG.model_copy(
            update={
                "predict_length": predict_length if predict_length is not None else workflow.model.max_horizon.value,
                "predict_context_length": predict_context_length,
            }
        )
        return cls(workflow=workflow, config=config, batch_size=batch_size)

    @property
    @override
    def quantiles(self) -> list[Quantile]:
        return self.workflow.model.quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """No-op: foundation models are zero-shot and need no per-window training."""

    def _build_window(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Build the context+horizon window for a single backtest event.

        Args:
            data: Time series with context, restricted to the event horizon.

        Returns:
            The window to forecast, or ``None`` when there is no observed target
            history before the horizon (no reliable forecast can be produced).
        """
        window = data.get_window(
            start=data.horizon - self.config.predict_context_length,
            end=data.horizon + self.config.predict_length,
            available_before=data.horizon,
        )

        target = window.data[self.workflow.model.target_column]
        if target[target.index < data.horizon].notna().sum() == 0:
            return None
        return window

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Forecast a single backtest window through the workflow.

        Returns:
            The workflow forecast for the window, or ``None`` when there is no
            observed target history before the horizon (no reliable forecast can
            be produced).
        """
        window = self._build_window(data)
        if window is None:
            return None
        return self.workflow.predict(data=window, forecast_start=data.horizon)

    @override
    def predict_batch(self, batch: list[RestrictedHorizonVersionedTimeSeries]) -> Sequence[TimeSeriesDataset | None]:
        """Forecast a batch of backtest windows in a single backend call.

        Each event becomes one window with its own forecast start; events with no
        usable target history are dropped and mapped back to ``None`` at their
        original index. The remaining windows are forecast with a single
        ``workflow.predict_batch`` call, then scattered back into input order.

        Args:
            batch: Backtest events to forecast.

        Returns:
            One forecast (or ``None``) per input event, in input order.
        """
        results: list[TimeSeriesDataset | None] = [None] * len(batch)
        prepared: list[tuple[int, TimeSeriesDataset, datetime]] = []
        for i, data in enumerate(batch):
            window = self._build_window(data)
            if window is not None:
                prepared.append((i, window, data.horizon))

        if not prepared:
            return results

        windows = [window for _, window, _ in prepared]
        starts = [start for _, _, start in prepared]
        forecasts = self.workflow.predict_batch(data=windows, forecast_start=starts)

        for (i, _, _), forecast in zip(prepared, forecasts, strict=True):
            results[i] = forecast
        return results


__all__ = [
    "DEFAULT_BACKTEST_CONFIG",
    "FoundationModelBacktestForecaster",
]
