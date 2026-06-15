
# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Chronos-2 foundation-model forecaster.

:class:`Chronos2Forecaster` adapts the generic
:class:`~openstef_foundation_models.inference.backend.InferenceBackend` to the
OpenSTEF :class:`~openstef_models.models.forecasting.forecaster.Forecaster`
contract. It owns the Chronos-2 specific pre- and post-processing while the
backend (ONNX or Torch) stays model-agnostic:

- **Preprocessing** builds the ``context``, ``attention_mask`` and ``group_ids``
  tensors from the raw target history. Chronos-2 normalises the context
  internally, so the raw load is fed unscaled.
- **Postprocessing** slices the model's frozen horizon to the requested length
  and resamples the model-native quantile grid onto the requested quantiles.

The model is zero-shot: there is nothing to train, so :meth:`fit` is a no-op and
:attr:`is_fitted` is always ``True`` once a backend is attached.
"""

from typing import ClassVar, override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins.predictor import BatchResult, HyperParams
from openstef_foundation_models.inference.backend import InferenceBackend
from openstef_foundation_models.utils.quantiles import interpolate_quantiles
from openstef_models.models.forecasting.forecaster import Forecaster


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

        Args:
            batch: Input datasets to forecast.

        Returns:
            One forecast dataset per input, in the same order.
        """
        inputs = self._build_inputs(batch)
        outputs = self.backend.run(inputs)
        predictions = np.asarray(outputs[self.backend.metadata.output_name])
        return [self._build_forecast(data, predictions[index]) for index, data in enumerate(batch)]

    def _build_inputs(self, batch: list[ForecastInputDataset]) -> dict[str, np.ndarray]:
        """Assemble the model input tensors for a batch of series.

        Args:
            batch: Input datasets to forecast.

        Returns:
            Mapping with ``context``, ``attention_mask`` and ``group_ids`` arrays.
        """
        context_length = self.backend.metadata.context_length
        contexts: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for data in batch:
            context, mask = self._build_context(data, context_length)
            contexts.append(context)
            masks.append(mask)
        return {
            "context": np.stack(contexts),
            "attention_mask": np.stack(masks),
            "group_ids": np.arange(len(batch), dtype=np.int64),
        }

    @staticmethod
    def _build_context(data: ForecastInputDataset, context_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Build the context and attention-mask row for a single series.

        Takes the most recent ``context_length`` target values strictly before
        the forecast start, left-padding with zeros when history is short.
        Missing values (padding or gaps) are zero-filled and flagged in the
        attention mask so the model ignores them.

        Args:
            data: Input dataset providing the target history.
            context_length: Number of context steps the model consumes.

        Returns:
            Tuple of ``(context, attention_mask)``, each of shape
            ``(context_length,)`` and dtype ``float32``.
        """
        forecast_start = pd.Timestamp(data.forecast_start)
        history = data.target_series
        values = history[history.index < forecast_start].to_numpy(dtype=np.float32)[-context_length:]

        padding = context_length - values.shape[0]
        if padding > 0:
            values = np.concatenate([np.full(padding, np.nan, dtype=np.float32), values])

        finite = np.isfinite(values)
        context = np.where(finite, values, np.float32(0.0)).astype(np.float32)
        mask = finite.astype(np.float32)
        return context, mask

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
