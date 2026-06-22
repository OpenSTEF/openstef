# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Chronos-2 foundation-model forecaster.

:class:`Chronos2Forecaster` adapts the generic
:class:`~openstef_foundation_models.inference.backend.InferenceBackend` to the
OpenSTEF :class:`~openstef_models.models.forecasting.forecaster.Forecaster`
contract. It owns the Chronos-2 specific pre- and post-processing while the
backend stays model-agnostic:

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

from typing import ClassVar, override

import numpy as np
import pandas as pd
from pydantic import Field

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.mixins.predictor import BatchResult, HyperParams
from openstef_core.utils.numpy import interpolate_quantiles, zero_fill_with_mask
from openstef_foundation_models.inference.backend import InferenceBackend
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

        Each series becomes a matrix of ``target + covariate`` rows over one
        ``context + horizon`` grid; the matrices are concatenated along the row
        (batch) axis and split once into the context and future blocks. Rows of a
        series share a ``group_id`` so Chronos-2 attends across the group. The
        target row's future is exactly what the model predicts, so it is blanked
        out.

        Args:
            batch: Input datasets to forecast.

        Returns:
            A tuple of the input mapping (``context``, ``attention_mask``,
            ``group_ids``, ``future_covariates``, ``future_covariates_mask``)
            and the output-row index of each series' target row.
        """
        context_length = self.backend.metadata.context_length
        horizon_length = self.backend.metadata.horizon_length
        max_covariates = self.backend.metadata.max_covariates

        matrices: list[np.ndarray] = []
        group_ids: list[int] = []
        target_indices: list[int] = []
        for group_id, data in enumerate(batch):
            target_indices.append(len(group_ids))
            matrix = self._build_group_matrix(data, context_length, horizon_length, max_covariates)
            matrices.append(matrix)
            group_ids.extend([group_id] * matrix.shape[0])

        values, mask = zero_fill_with_mask(np.concatenate(matrices))

        # The target row's future is exactly what the model predicts: blank it out.
        values[target_indices, context_length:] = np.float32(0.0)
        mask[target_indices, context_length:] = np.float32(0.0)

        inputs = {
            "context": values[:, :context_length],
            "attention_mask": mask[:, :context_length],
            "group_ids": np.asarray(group_ids, dtype=np.int64),
            "future_covariates": values[:, context_length:],
            "future_covariates_mask": mask[:, context_length:],
        }
        return inputs, target_indices

    @staticmethod
    def _build_group_matrix(
        data: ForecastInputDataset,
        context_length: int,
        horizon_length: int,
        max_covariates: int | None,
    ) -> np.ndarray:
        """Build the ``target + covariate`` matrix for a single series group.

        The target and every covariate column are reindexed onto one grid that
        spans the context window and the forecast horizon. Row 0 is the target;
        each remaining row is a covariate. Missing timestamps stay ``NaN`` here -
        the caller turns them into zeros plus a mask.

        When ``max_covariates`` is set the checkpoint froze its covariate axis, so
        the row count must be exactly ``1 + max_covariates``: a series with fewer
        covariates is padded with all-``NaN`` rows (masked out downstream, so they
        do not affect the forecast) and one with more is rejected, turning what
        would be an opaque ONNX Runtime shape error into a clear message.

        Args:
            data: Input dataset providing the target and covariate columns.
            context_length: Number of context steps the model consumes.
            horizon_length: Frozen forecast horizon the model emits.
            max_covariates: Frozen covariate-row count, or ``None`` when the
                covariate axis is dynamic and no padding is needed.

        Returns:
            Matrix of shape ``(n_rows, context_length + horizon_length)``, target
            row first, where ``n_rows`` is ``1 + max_covariates`` when the axis is
            frozen.

        Raises:
            ValueError: If the series has more covariates than the frozen
                ``max_covariates`` axis can hold.
        """
        columns = [data.target_column, *(name for name in data.feature_names if name != data.target_column)]
        forecast_start = pd.Timestamp(data.forecast_start)
        index = pd.date_range(
            start=forecast_start - context_length * data.sample_interval,
            periods=context_length + horizon_length,
            freq=data.sample_interval,
        )
        matrix = data.data[columns].reindex(index).to_numpy(dtype=np.float32).T
        if max_covariates is None:
            return matrix

        n_covariates = matrix.shape[0] - 1
        if n_covariates > max_covariates:
            msg = (
                f"Series has {n_covariates} covariates but the checkpoint's covariate axis is frozen at "
                f"{max_covariates}; drop covariates or use a dynamic-shape checkpoint."
            )
            raise ValueError(msg)
        if n_covariates < max_covariates:
            padding = np.full((max_covariates - n_covariates, matrix.shape[1]), np.nan, dtype=np.float32)
            matrix = np.concatenate([matrix, padding])
        return matrix

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
