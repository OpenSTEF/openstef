# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Backtesting adapter that runs a forecasting workflow through openstef-beam.

Requires the ``benchmarking`` extra::

    pip install openstef-foundation-models[benchmarking]

The adapter bridges beam's backtesting interface
(:class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin`)
to an OpenSTEF
:class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow`.
It wraps a **single, already-built** workflow instance and reuses it across every
backtest window, so an expensive backend (e.g. a loaded ONNX session) is created
once and shared — there is no per-window model loading. Every window is forecast
by calling the workflow's own :meth:`~CustomForecastingWorkflow.predict`, so the
model's preprocessing (feature selection / covariates) and postprocessing
(quantile sorting) apply uniformly.

Forecasting runs one window at a time. Batching multiple windows into a single
backend call is a separate, planned optimisation and is intentionally not done
here.

Foundation models such as Chronos-2 are zero-shot, so :meth:`fit` is a no-op and
``requires_training`` defaults to ``False``.
"""

from datetime import timedelta
from typing import override

from pydantic import Field

from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import MissingExtraError
from openstef_core.types import Quantile
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

try:
    from openstef_beam.backtesting.backtest_forecaster.mixins import (
        BacktestForecasterConfig,
        BacktestForecasterMixin,
    )
    from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
except ImportError as e:
    raise MissingExtraError("openstef-beam", "openstef-foundation-models") from e

#: Default amount of history fed as model context. Foundation models truncate the
#: context to their own window internally, so a generous default is harmless.
_DEFAULT_CONTEXT_LENGTH = timedelta(days=60)


class FoundationModelBacktestForecaster(BaseModel, BacktestForecasterMixin):
    """Backtest wrapper around a single, shared forecasting workflow.

    The wrapped :attr:`workflow` is built once and reused for every prediction
    window (load-once). Each window is forecast through
    :meth:`~CustomForecastingWorkflow.predict` with the window horizon as the
    forecast start, so the adapter never reaches into the workflow's model.
    """

    workflow: CustomForecastingWorkflow = Field(
        description="The shared, pre-built forecasting workflow to run for every window."
    )
    config: BacktestForecasterConfig = Field(description="Backtest window configuration.")

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
            end=data.horizon,
            available_before=data.horizon,
        )

        target = window.data[self.workflow.model.target_column]
        if target[target.index < data.horizon].notna().sum() == 0:
            return None

        return self.workflow.predict(data=window, forecast_start=data.horizon)


def create_foundation_model_backtest_forecaster(
    workflow: CustomForecastingWorkflow,
    *,
    predict_length: timedelta | None = None,
    predict_context_length: timedelta = _DEFAULT_CONTEXT_LENGTH,
    config: BacktestForecasterConfig | None = None,
) -> FoundationModelBacktestForecaster:
    """Wrap a forecasting workflow for backtesting through openstef-beam.

    Builds a load-once backtest forecaster that reuses *workflow* (and its
    backend) across every window.

    Args:
        workflow: A pre-built forecasting workflow to run for every window.
        predict_length: Forecast horizon per window. Defaults to the workflow's
            maximum configured horizon.
        predict_context_length: Amount of history fed as model context.
        config: Full backtest config. When given, it overrides the derived
            window settings (``predict_length``/``predict_context_length``).

    Returns:
        A configured :class:`FoundationModelBacktestForecaster`.
    """
    if config is None:
        config = BacktestForecasterConfig(
            requires_training=False,
            predict_length=predict_length if predict_length is not None else workflow.model.max_horizon.value,
            predict_min_length=timedelta(minutes=15),
            predict_context_length=predict_context_length,
            predict_context_min_coverage=0.0,
            training_context_length=timedelta(0),
            training_context_min_coverage=0.0,
        )

    return FoundationModelBacktestForecaster(
        workflow=workflow,
        config=config,
    )


__all__ = [
    "FoundationModelBacktestForecaster",
    "create_foundation_model_backtest_forecaster",
]
