
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Chronos-2 foundation-model forecaster.

:class:`Chronos2Forecaster` adapts the generic
:class:`~openstef_foundation_models.inference.backend.InferenceBackend` to the
OpenSTEF :class:`~openstef_models.models.forecasting.forecaster.Forecaster`
contract. It owns the Chronos-2 specific pre- and post-processing while the
backend (ONNX or Torch) stays model-agnostic:

- **Preprocessing** builds the ``context``, ``attention_mask``, ``group_ids``,
  ``future_covariates`` and ``future_covariates_mask`` tensors. Every non-target
  feature column is treated as a *known* covariate: its history feeds an extra
  context row and its horizon values feed ``future_covariates``. Chronos-2
  shares attention within a group, so the target series and its covariates share
  one ``group_id``. Chronos-2 normalises each series internally, so raw values
  are fed unscaled.
- **Postprocessing** picks each series' target row out of the batched output,
  slices the model's frozen horizon to the requested length and resamples the
  model-native quantile grid onto the requested quantiles.

The model is zero-shot: there is nothing to train, so :meth:`fit` is a no-op and
:attr:`is_fitted` is always ``True`` once a backend is attached.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins.predictor import BatchResult, HyperParams
from openstef_core.utils.quantiles import interpolate_quantiles
from openstef_foundation_models.inference.backend import InferenceBackend
from openstef_models.models.forecasting.forecaster import Forecaster


@dataclass(frozen=True)
class _SeriesRows:
    """Assembled model rows for a single series and its covariates.

    A series contributes one target row followed by one row per covariate. All
    rows of a series share a ``group_id`` so Chronos-2 attends across them. The
    target row's ``future_covariates`` are masked out (its future is unknown);
    covariate rows carry their known horizon values.
    """

    context: np.ndarray
    """Context windows, shape ``(n_rows, context_length)``; row 0 is the target."""
    attention_mask: np.ndarray
    """Validity mask for ``context``, same shape."""
    future_covariates: np.ndarray
    """Horizon covariate values, shape ``(n_rows, horizon_length)``."""
    future_covariates_mask: np.ndarray
    """Known-value mask for ``future_covariates``, same shape."""


class Chronos2HyperParams(HyperParams):
    """Hyperparameters for :class:`Chronos2Forecaster`.

    Chronos-2 is a pretrained zero-shot model, so it exposes no trainable or
    tunable hyperparameters. The class exists to satisfy the forecaster contract
    and to host future inference-time knobs.
    """


class Chronos2Forecaster(Forecaster):
    """Zero-shot probabilistic forecaster backed by a Chronos-2 checkpoint.

    The forecaster composes an :class:`InferenceBackend` (built once and reused
    across an entire backtest) and translates between OpenSTEF datasets and the
    model's tensor interface. Prediction is batch-first: :meth:`predict_batch`
    runs the backend once over a stack of series and :meth:`predict` is a
    batch-of-one wrapper.
    """

    HyperParams: ClassVar[type[Chronos2HyperParams]] = Chronos2HyperParams

    backend: InferenceBackend = Field(
        description="Execution backend wrapping the resolved Chronos-2 checkpoint.",
    )
    hyperparams: Chronos2HyperParams = Field(
        default_factory=Chronos2HyperParams,
        description="Inference hyperparameters (none are tunable for Chronos-2).",
    )
    supports_batching: bool = Field(
        default=True,
        description="Chronos-2 runs a whole batch of series in a single backend call.",
    )

    @property
    @override
    def hparams(self) -> Chronos2HyperParams:
        return self.hyperparams

    @property
    @override
    def is_fitted(self) -> bool:
        return True

    @override
    def fit(self, data: ForecastInputDataset, data_val: ForecastInputDataset | None = None) -> None:
        """Fit the forecaster.

        Chronos-2 is pretrained and zero-shot, so there is nothing to fit. The
        method exists only to satisfy the forecaster contract.

        Args:
            data: Unused training data.
            data_val: Unused validation data.
        """

    @override
    def predict(self, data: ForecastInputDataset) -> ForecastDataset:
        """Forecast a single series.

        Args:
            data: Input dataset whose target history provides the model context.

        Returns:
            Probabilistic forecast for the requested quantiles and horizon.
        """
        return self._forecast([data])[0]

    @override
    def predict_batch(self, data: list[ForecastInputDataset]) -> BatchResult[ForecastDataset]:
        """Forecast a batch of series in a single backend call.

        Args:
            data: Input datasets to forecast. Each provides its own target
                history and forecast start.

        Returns:
            One forecast per input dataset, in the same order.
        """
        results: BatchResult[ForecastDataset] = []
        results.extend(self._forecast(data))
        return results

    def _forecast(self, batch: list[ForecastInputDataset]) -> list[ForecastDataset]:
        """Run the backend once over *batch* and post-process each forecast.

        Each series contributes a target row plus one row per covariate, all
        sharing a ``group_id``. After inference, each series' target row is
        sliced back out of the batched output.

        Args:
            batch: Input datasets to forecast.

        Returns:
            One forecast dataset per input, in the same order.
        """
        inputs, target_indices = self._build_inputs(batch)
        outputs = self.backend.run(inputs)
        predictions = np.asarray(outputs[self.backend.metadata.output_name])
        return [
            self._build_forecast(data, predictions[target_index])
            for data, target_index in zip(batch, target_indices, strict=True)
        ]

    def _build_inputs(self, batch: list[ForecastInputDataset]) -> tuple[dict[str, np.ndarray], list[int]]:
        """Assemble the batched model input tensors for a batch of series.

        Rows are laid out series by series: for each series its target row comes
        first, followed by one row per covariate. All rows of a series share a
        ``group_id`` so Chronos-2 attends across the group.

        Args:
            batch: Input datasets to forecast.

        Returns:
            A tuple of the input mapping (``context``, ``attention_mask``,
            ``group_ids``, ``future_covariates``, ``future_covariates_mask``)
            and the output-row index of each series' target row.
        """
        context_length = self.backend.metadata.context_length
        horizon_length = self.backend.metadata.horizon_length

        contexts: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        future_covariates: list[np.ndarray] = []
        future_masks: list[np.ndarray] = []
        group_ids: list[int] = []
        target_indices: list[int] = []

        for group_id, data in enumerate(batch):
            target_indices.append(len(contexts))
            rows = self._build_series_rows(data, context_length, horizon_length)
            n_rows = rows.context.shape[0]
            contexts.extend(rows.context)
            masks.extend(rows.attention_mask)
            future_covariates.extend(rows.future_covariates)
            future_masks.extend(rows.future_covariates_mask)
            group_ids.extend([group_id] * n_rows)

        inputs = {
            "context": np.stack(contexts),
            "attention_mask": np.stack(masks),
            "group_ids": np.asarray(group_ids, dtype=np.int64),
            "future_covariates": np.stack(future_covariates),
            "future_covariates_mask": np.stack(future_masks),
        }
        return inputs, target_indices

    def _build_series_rows(
        self,
        data: ForecastInputDataset,
        context_length: int,
        horizon_length: int,
    ) -> _SeriesRows:
        """Build the target and covariate rows for a single series.

        The target row carries the target history in ``context`` and an empty
        (masked-out) future. Each non-target feature column becomes a covariate
        row carrying its own history in ``context`` and its known horizon values
        in ``future_covariates``.

        Args:
            data: Input dataset providing the target and covariate columns.
            context_length: Number of context steps the model consumes.
            horizon_length: Frozen forecast horizon the model emits.

        Returns:
            The assembled rows for this series, target first.
        """
        target_context, target_mask = self._build_context(data.target_series, data.forecast_start, context_length)
        contexts = [target_context]
        masks = [target_mask]
        future_covariates = [np.zeros(horizon_length, dtype=np.float32)]
        future_masks = [np.zeros(horizon_length, dtype=np.float32)]

        horizon_index = pd.date_range(
            start=pd.Timestamp(data.forecast_start),
            periods=horizon_length,
            freq=data.sample_interval,
        )
        covariate_columns = [name for name in data.feature_names if name != data.target_column]
        for name in covariate_columns:
            series = data.data[name]
            cov_context, cov_mask = self._build_context(series, data.forecast_start, context_length)
            cov_future, cov_future_mask = self._build_future_covariate(series, horizon_index)
            contexts.append(cov_context)
            masks.append(cov_mask)
            future_covariates.append(cov_future)
            future_masks.append(cov_future_mask)

        return _SeriesRows(
            context=np.stack(contexts),
            attention_mask=np.stack(masks),
            future_covariates=np.stack(future_covariates),
            future_covariates_mask=np.stack(future_masks),
        )

    @staticmethod
    def _build_context(
        series: pd.Series,
        forecast_start: datetime,
        context_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the context and attention-mask row for a single series.

        Takes the most recent ``context_length`` values strictly before the
        forecast start, left-padding with zeros when history is short. Missing
        values (padding or gaps) are zero-filled and flagged in the attention
        mask so the model ignores them.

        Args:
            series: Time-indexed values (target or covariate).
            forecast_start: Timestamp marking the first forecast step.
            context_length: Number of context steps the model consumes.

        Returns:
            Tuple of ``(context, attention_mask)``, each of shape
            ``(context_length,)`` and dtype ``float32``.
        """
        start = pd.Timestamp(forecast_start)
        values = series[series.index < start].to_numpy(dtype=np.float32)[-context_length:]

        padding = context_length - values.shape[0]
        if padding > 0:
            values = np.concatenate([np.full(padding, np.nan, dtype=np.float32), values])

        finite = np.isfinite(values)
        context = np.where(finite, values, np.float32(0.0)).astype(np.float32)
        mask = finite.astype(np.float32)
        return context, mask

    @staticmethod
    def _build_future_covariate(
        series: pd.Series,
        horizon_index: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the known-future covariate row for a single covariate column.

        The covariate is reindexed onto the model's frozen horizon. Steps with a
        known value are passed through with mask 1; steps that fall outside the
        covariate's range (or are NaN) are zero-filled with mask 0 so the model
        ignores them.

        Args:
            series: Time-indexed covariate values spanning history and future.
            horizon_index: Timestamps of the model's frozen horizon.

        Returns:
            Tuple of ``(future_covariates, future_covariates_mask)``, each of
            shape ``(len(horizon_index),)`` and dtype ``float32``.
        """
        values = series.reindex(horizon_index).to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        future = np.where(finite, values, np.float32(0.0)).astype(np.float32)
        mask = finite.astype(np.float32)
        return future, mask

    def _build_forecast(self, data: ForecastInputDataset, predictions: np.ndarray) -> ForecastDataset:
        """Post-process one series' raw quantile predictions into a dataset.

        Args:
            data: Input dataset the prediction was produced for.
            predictions: Raw model output of shape ``(n_native_quantiles, horizon)``.

        Returns:
            Forecast dataset sliced to the requested horizon and resampled onto
            the requested quantiles.
        """
        native = predictions.T  # (horizon, n_native_quantiles)

        forecast_index = data.create_forecast_range(self.max_horizon)
        steps = min(len(forecast_index), native.shape[0])
        forecast_index = forecast_index[:steps]

        resampled = interpolate_quantiles(
            native[:steps],
            self.backend.metadata.native_quantiles,
            self.quantiles,
        )
        return ForecastDataset.from_quantile_predictions(
            predictions=resampled,
            index=forecast_index,
            quantiles=self.quantiles,
            sample_interval=data.sample_interval,
            target_column=data.target_column,
        )


__all__ = [
    "Chronos2Forecaster",
    "Chronos2HyperParams",
]
