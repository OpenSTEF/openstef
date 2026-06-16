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
once and shared — there is no per-window model loading. Running through the
workflow means the model's preprocessing (feature selection / covariates) and
postprocessing (quantile sorting) apply to every window.

Foundation models such as Chronos-2 are zero-shot, so :meth:`fit` is a no-op and
``requires_training`` defaults to ``False``.
"""

import warnings
from collections.abc import Sequence
from datetime import timedelta
from typing import Self, override

from pydantic import Field, model_validator

from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.exceptions import MissingExtraError
from openstef_core.types import Quantile
from openstef_models.models.forecasting_model import ForecastingModel, restore_target
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow

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
    """Backtest wrapper around a single, shared forecasting workflow.

    The wrapped :attr:`workflow` is built once and reused for every prediction
    window (load-once). Each window is translated into a
    :class:`~openstef_core.datasets.validated_datasets.ForecastInputDataset`
    whose ``forecast_start`` is the window horizon (after the workflow's
    preprocessing runs), and the forecaster's batch path is used so a whole batch
    of windows runs in a single backend call.
    """

    workflow: CustomForecastingWorkflow = Field(
        description="The shared, pre-built forecasting workflow to run for every window."
    )
    config: BacktestForecasterConfig = Field(description="Backtest window configuration.")
    batch_size: int | None = Field(
        default=None,
        description="Maximum windows per backend call. None or 1 disables batching.",
    )

    @model_validator(mode="after")
    def _validate_template_model(self) -> Self:
        """Ensure the workflow wraps a single-forecaster model the adapter can drive.

        Returns:
            The validated adapter.

        Raises:
            TypeError: If the workflow model is not a :class:`ForecastingModel`.
        """
        if not isinstance(self.workflow.model, ForecastingModel):
            msg = (
                "FoundationModelBacktestForecaster requires a workflow wrapping a ForecastingModel, "
                f"got {type(self.workflow.model).__name__}."
            )
            raise TypeError(msg)
        return self

    @property
    def _model(self) -> ForecastingModel:
        """The wrapped single-forecaster model (validated at construction)."""
        model = self.workflow.model
        assert isinstance(model, ForecastingModel)  # noqa: S101 - guaranteed by validator
        return model

    @property
    @override
    def quantiles(self) -> list[Quantile]:
        return self._model.quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """No-op: foundation models are zero-shot and need no per-window training."""

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        input_dataset = self._to_input(data)
        if input_dataset is None:
            return None
        raw_forecast = self._model.forecaster.predict(input_dataset)
        return self._finalize(raw_forecast, input_dataset)

    @override
    def predict_batch(
        self, batch: list[RestrictedHorizonVersionedTimeSeries]
    ) -> Sequence[TimeSeriesDataset | None]:
        results: list[TimeSeriesDataset | None] = [None] * len(batch)

        prepared: list[tuple[int, ForecastInputDataset]] = []
        for position, data in enumerate(batch):
            input_dataset = self._to_input(data)
            if input_dataset is not None:
                prepared.append((position, input_dataset))
        if not prepared:
            return results

        raw_forecasts = self._model.forecaster.predict_batch([dataset for _, dataset in prepared])
        for (position, input_dataset), raw_forecast in zip(prepared, raw_forecasts, strict=True):
            results[position] = self._finalize(raw_forecast, input_dataset)
        return results

    def _finalize(self, raw_forecast: ForecastDataset, input_dataset: ForecastInputDataset) -> TimeSeriesDataset:
        """Apply the same target restoration and postprocessing as the workflow.

        Mirrors :meth:`ForecastingModel.predict` so the batched path produces the
        same output as a single :meth:`predict`.

        Returns:
            The postprocessed forecast.
        """
        model = self._model
        restored = restore_target(
            dataset=raw_forecast, original_dataset=input_dataset, target_column=model.target_column
        )
        return model.postprocessing.transform(data=restored)

    def _to_input(self, data: RestrictedHorizonVersionedTimeSeries) -> ForecastInputDataset | None:
        """Translate a backtest window into a preprocessed forecaster input dataset.

        Returns:
            The input dataset, or ``None`` when there is no observed target
            history before the horizon (no reliable forecast can be produced).
        """
        model = self._model
        window = data.get_window(
            start=data.horizon - self.config.predict_context_length,
            end=data.horizon,
            available_before=data.horizon,
        )
        input_dataset = model.prepare_input(data=window, forecast_start=data.horizon)

        history = input_dataset.target_series
        if history[history.index < data.horizon].notna().sum() == 0:
            return None
        return input_dataset


def create_foundation_model_backtest_forecaster(
    workflow: CustomForecastingWorkflow,
    *,
    predict_length: timedelta | None = None,
    predict_context_length: timedelta = _DEFAULT_CONTEXT_LENGTH,
    batch_size: int | None = None,
    config: BacktestForecasterConfig | None = None,
) -> FoundationModelBacktestForecaster:
    """Wrap a forecasting workflow for backtesting through openstef-beam.

    Builds a load-once backtest forecaster that reuses *workflow* (and its
    backend) across every window.

    Args:
        workflow: A pre-built forecasting workflow to run for every window.
        predict_length: Forecast horizon per window. Defaults to the workflow
            model's maximum configured horizon.
        predict_context_length: Amount of history fed as model context.
        batch_size: Maximum windows per backend call. None or 1 disables batching.
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
        batch_size=batch_size,
    )


__all__ = [
    "FoundationModelBacktestForecaster",
    "create_foundation_model_backtest_forecaster",
]
