# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Backtesting adapter that runs any :class:`Forecaster` through openstef-beam.

Requires the ``benchmarking`` extra::

    pip install openstef-foundation-models[benchmarking]

The adapter bridges beam's backtesting interface
(:class:`~openstef_beam.backtesting.backtest_forecaster.mixins.BacktestForecasterMixin`)
to the OpenSTEF :class:`~openstef_models.models.forecasting.forecaster.Forecaster`
contract. It wraps a **single, already-built** forecaster instance and reuses it
across every backtest window, so an expensive backend (e.g. a loaded ONNX
session) is created once and shared — there is no per-window model loading.

Foundation models such as Chronos-2 are zero-shot, so :meth:`fit` is a no-op and
``requires_training`` defaults to ``False``.
"""

import warnings
from collections.abc import Sequence
from datetime import timedelta
from typing import override

from pydantic import Field

from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.exceptions import MissingExtraError
from openstef_core.types import Quantile
from openstef_models.models.forecasting.forecaster import Forecaster

try:
    from openstef_beam.backtesting.backtest_forecaster.mixins import (
        BacktestBatchForecasterMixin,
        BacktestForecasterConfig,
        BacktestForecasterMixin,
    )
    from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
except ImportError as e:
    raise MissingExtraError("openstef-beam", "openstef-foundation-models") from e

#: Default amount of history fed as model context. Foundation models truncate the
#: context to their own window internally, so a generous default is harmless.
_DEFAULT_CONTEXT_LENGTH = timedelta(days=60)

# BacktestBatchForecasterMixin assigns ``batch_size`` a pydantic FieldInfo as a plain-class
# attribute. Redeclaring it as a field below (so it serialises and validates) makes pydantic warn
# that the field shadows a parent attribute. The redeclaration is intentional, so silence that one
# precise message rather than leaking a benign warning on every import.
warnings.filterwarnings(
    "ignore",
    message='Field name "batch_size" in "FoundationModelBacktestForecaster" shadows an attribute in parent',
    category=UserWarning,
)


class FoundationModelBacktestForecaster(BaseModel, BacktestBatchForecasterMixin, BacktestForecasterMixin):
    """Backtest wrapper around a single, shared foundation-model forecaster.

    The wrapped :attr:`forecaster` is built once and reused for every prediction
    window (load-once). Each window is translated into a
    :class:`~openstef_core.datasets.validated_datasets.ForecastInputDataset`
    whose ``forecast_start`` is the window horizon, and the forecaster's batch
    path is used so a whole batch of windows runs in a single backend call.
    """

    forecaster: Forecaster = Field(description="The shared, pre-built forecaster to run for every window.")
    config: BacktestForecasterConfig = Field(description="Backtest window configuration.")
    target_column: str = Field(default="load", description="Name of the target column in the backtest data.")
    batch_size: int | None = Field(
        default=None,
        description="Maximum windows per backend call. None or 1 disables batching.",
    )

    @property
    @override
    def quantiles(self) -> list[Quantile]:
        return self.forecaster.quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """No-op: foundation models are zero-shot and need no per-window training."""

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        input_dataset = self._to_input(data)
        if input_dataset is None:
            return None
        return self.forecaster.predict(input_dataset)

    @override
    def predict_batch(
        self, batch: list[RestrictedHorizonVersionedTimeSeries]
    ) -> Sequence[TimeSeriesDataset | None]:
        inputs = [self._to_input(data) for data in batch]
        results: list[TimeSeriesDataset | None] = [None] * len(batch)

        valid = [(position, dataset) for position, dataset in enumerate(inputs) if dataset is not None]
        if not valid:
            return results

        forecasts = self.forecaster.predict_batch([dataset for _, dataset in valid])
        for (position, _), forecast in zip(valid, forecasts, strict=True):
            results[position] = forecast
        return results

    def _to_input(self, data: RestrictedHorizonVersionedTimeSeries) -> ForecastInputDataset | None:
        """Translate a backtest window into a forecaster input dataset.

        Returns:
            The input dataset, or ``None`` when there is no observed target
            history before the horizon (no reliable forecast can be produced).
        """
        window = data.get_window(
            start=data.horizon - self.config.predict_context_length,
            end=data.horizon,
            available_before=data.horizon,
        )
        input_dataset = ForecastInputDataset.from_timeseries(
            window,
            target_column=self.target_column,
            forecast_start=data.horizon,
        )

        history = input_dataset.target_series
        if history[history.index < data.horizon].notna().sum() == 0:
            return None
        return input_dataset


def create_foundation_model_backtest_forecaster(
    forecaster: Forecaster,
    *,
    predict_length: timedelta | None = None,
    predict_context_length: timedelta = _DEFAULT_CONTEXT_LENGTH,
    batch_size: int | None = None,
    target_column: str = "load",
    config: BacktestForecasterConfig | None = None,
) -> FoundationModelBacktestForecaster:
    """Wrap a forecaster for backtesting through openstef-beam.

    Builds a load-once backtest forecaster that reuses *forecaster* (and its
    backend) across every window.

    Args:
        forecaster: A pre-built forecaster to run for every window.
        predict_length: Forecast horizon per window. Defaults to the forecaster's
            maximum configured horizon.
        predict_context_length: Amount of history fed as model context.
        batch_size: Maximum windows per backend call. None or 1 disables batching.
        target_column: Name of the target column in the backtest data.
        config: Full backtest config. When given, it overrides the derived
            window settings (``predict_length``/``predict_context_length``).

    Returns:
        A configured :class:`FoundationModelBacktestForecaster`.
    """
    if config is None:
        config = BacktestForecasterConfig(
            requires_training=False,
            predict_length=predict_length if predict_length is not None else forecaster.max_horizon.value,
            predict_min_length=timedelta(minutes=15),
            predict_context_length=predict_context_length,
            predict_context_min_coverage=0.0,
            training_context_length=timedelta(0),
            training_context_min_coverage=0.0,
        )

    return FoundationModelBacktestForecaster(
        forecaster=forecaster,
        config=config,
        target_column=target_column,
        batch_size=batch_size,
    )


__all__ = [
    "FoundationModelBacktestForecaster",
    "create_foundation_model_backtest_forecaster",
]
