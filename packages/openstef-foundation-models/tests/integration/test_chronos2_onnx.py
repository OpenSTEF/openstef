# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Integration tests running the real exported Chronos-2 ONNX checkpoint.

These exercise the full ONNX path end to end — the backend, the forecaster's
pre-/post-processing, and quantile resampling — against the actual artifact.
They are marked ``slow`` + ``integration`` and skip when the artifact is absent.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.inference.onnx_backend import OnnxBackend
from openstef_foundation_models.models.checkpoint import CheckpointMetadata
from openstef_foundation_models.models.forecasting import Chronos2Forecaster

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SAMPLE_INTERVAL = timedelta(minutes=15)
N_NATIVE_QUANTILES = 21
HORIZON_LENGTH = 672
QUANTILES = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]


def _seasonal_load(periods: int, *, start: str = "2025-01-01") -> pd.DataFrame:
    """Build a synthetic load series with a daily sine pattern and positive level."""
    index = pd.date_range(start, periods=periods, freq=SAMPLE_INTERVAL)
    steps_per_day = int(timedelta(days=1) / SAMPLE_INTERVAL)
    phase = 2 * np.pi * np.arange(periods) / steps_per_day
    load = 100.0 + 30.0 * np.sin(phase)
    return pd.DataFrame({"load": load}, index=index)


def _make_input(periods: int, horizon: LeadTime, *, start: str = "2025-01-01") -> ForecastInputDataset:
    """Build a forecast input whose forecast starts right after the history."""
    frame = _seasonal_load(periods, start=start)
    return ForecastInputDataset(
        data=frame,
        sample_interval=SAMPLE_INTERVAL,
        target_column="load",
        forecast_start=frame.index[-1].to_pydatetime() + SAMPLE_INTERVAL,
    )


def _make_input_with_covariate(periods: int, *, start: str = "2025-01-01") -> ForecastInputDataset:
    """Build a forecast input with a known covariate spanning history and the full horizon.

    The target is observed only over the history; the covariate column is known
    over both the history and the entire frozen horizon (so it is forwarded as a
    fully-known future covariate).
    """
    total = periods + HORIZON_LENGTH
    index = pd.date_range(start, periods=total, freq=SAMPLE_INTERVAL)
    steps_per_day = int(timedelta(days=1) / SAMPLE_INTERVAL)
    phase = 2 * np.pi * np.arange(total) / steps_per_day

    load = 100.0 + 30.0 * np.sin(phase)
    load[periods:] = np.nan  # target is unknown over the horizon
    covariate = 50.0 + 20.0 * np.sin(phase + np.pi / 4)  # known everywhere

    frame = pd.DataFrame({"load": load, "temperature_2m": covariate}, index=index)
    return ForecastInputDataset(
        data=frame,
        sample_interval=SAMPLE_INTERVAL,
        target_column="load",
        forecast_start=index[periods].to_pydatetime(),
    )


def test_onnx_backend_produces_native_quantile_shape(onnx_backend: OnnxBackend) -> None:
    """The raw backend returns finite native-quantile predictions of the frozen shape."""
    # Arrange
    batch_size, context_length = 2, 512
    rng = np.random.default_rng(0)
    inputs = {
        "context": rng.normal(100.0, 10.0, size=(batch_size, context_length)).astype(np.float32),
        "group_ids": np.arange(batch_size, dtype=np.int64),
        "attention_mask": np.ones((batch_size, context_length), dtype=np.float32),
        # Target-only path: covariates masked out (the horizon length is frozen at 672).
        "future_covariates": np.zeros((batch_size, HORIZON_LENGTH), dtype=np.float32),
        "future_covariates_mask": np.zeros((batch_size, HORIZON_LENGTH), dtype=np.float32),
    }

    # Act
    outputs = onnx_backend.run(inputs)

    # Assert
    predictions = np.asarray(outputs["quantile_preds"])
    assert predictions.shape == (batch_size, N_NATIVE_QUANTILES, HORIZON_LENGTH)
    assert np.isfinite(predictions).all()


def test_onnx_backend_runs_with_known_future_covariates(onnx_backend: OnnxBackend) -> None:
    """A grouped target+covariate batch with known-future values runs and stays finite."""
    # Arrange: row 0 is the target, row 1 is a covariate sharing its group id.
    context_length = 512
    rng = np.random.default_rng(1)
    context = rng.normal(100.0, 10.0, size=(2, context_length)).astype(np.float32)
    inputs = {
        "context": context,
        "group_ids": np.array([0, 0], dtype=np.int64),
        "attention_mask": np.ones((2, context_length), dtype=np.float32),
        # Target row (0): covariates masked out. Covariate row (1): known horizon values.
        "future_covariates": np.vstack(
            [
                np.zeros(HORIZON_LENGTH, dtype=np.float32),
                rng.normal(50.0, 5.0, size=HORIZON_LENGTH).astype(np.float32),
            ]
        ),
        "future_covariates_mask": np.vstack(
            [
                np.zeros(HORIZON_LENGTH, dtype=np.float32),
                np.ones(HORIZON_LENGTH, dtype=np.float32),
            ]
        ),
    }

    # Act
    predictions = np.asarray(onnx_backend.run(inputs)["quantile_preds"])

    # Assert
    assert predictions.shape == (2, N_NATIVE_QUANTILES, HORIZON_LENGTH)
    assert np.isfinite(predictions).all()


def test_chronos2_forecaster_produces_valid_probabilistic_forecast(onnx_backend: OnnxBackend) -> None:
    """End to end: a positive, horizon-sized, quantile-monotone raw-scale forecast."""
    # Arrange
    horizon = LeadTime.from_string("PT24H")
    forecaster = Chronos2Forecaster(backend=onnx_backend, quantiles=QUANTILES, horizons=[horizon])
    data = _make_input(periods=14 * 96, horizon=horizon)
    expected_steps = len(data.create_forecast_range(horizon))

    # Act
    forecast = forecaster.predict(data)

    # Assert
    quantiles_data = forecast.quantiles_data
    assert list(forecast.quantiles) == QUANTILES
    assert len(forecast.data) == expected_steps
    assert forecast.data.index[0].to_pydatetime() == data.forecast_start
    # Forecast is on the raw load scale (model owns normalization), so values stay positive.
    assert (quantiles_data.to_numpy() > 0).all()
    # Quantiles must be non-decreasing across levels at every timestep.
    p10 = quantiles_data[Quantile(0.1).format()].to_numpy()
    p50 = quantiles_data[Quantile(0.5).format()].to_numpy()
    p90 = quantiles_data[Quantile(0.9).format()].to_numpy()
    assert np.all(p10 <= p50 + 1e-6)
    assert np.all(p50 <= p90 + 1e-6)


def test_chronos2_predict_batch_matches_individual_predicts(onnx_backend: OnnxBackend) -> None:
    """A batched call yields the same forecasts as predicting each series alone."""
    # Arrange
    horizon = LeadTime.from_string("PT12H")
    forecaster = Chronos2Forecaster(backend=onnx_backend, quantiles=QUANTILES, horizons=[horizon])
    batch = [
        _make_input(periods=7 * 96, horizon=horizon, start="2025-01-01"),
        _make_input(periods=10 * 96, horizon=horizon, start="2025-02-01"),
    ]

    # Act
    batched = forecaster.predict_batch(batch)
    individual = [forecaster.predict(data) for data in batch]

    # Assert
    assert len(batched) == len(batch)
    for batched_forecast, single_forecast in zip(batched, individual, strict=True):
        np.testing.assert_allclose(
            batched_forecast.quantiles_data.to_numpy(),
            single_forecast.quantiles_data.to_numpy(),
            rtol=1e-4,
            atol=1e-4,
        )


def test_onnx_backend_metadata_matches_checkpoint(
    onnx_backend: OnnxBackend, chronos2_metadata: CheckpointMetadata
) -> None:
    """The backend exposes the resolved checkpoint metadata for the forecaster to use."""
    # Assert
    assert onnx_backend.metadata.model_family == chronos2_metadata.model_family
    assert onnx_backend.metadata.horizon_length == HORIZON_LENGTH
    assert onnx_backend.metadata.native_quantiles == chronos2_metadata.native_quantiles


def test_chronos2_forecaster_conditions_on_covariates(onnx_backend: OnnxBackend) -> None:
    """A known covariate conditions the forecast: it differs from the target-only run."""
    # Arrange
    horizon = LeadTime.from_string("PT24H")
    forecaster = Chronos2Forecaster(backend=onnx_backend, quantiles=QUANTILES, horizons=[horizon])
    with_covariate = _make_input_with_covariate(periods=14 * 96)
    without_covariate = ForecastInputDataset(
        data=with_covariate.data[["load"]],
        sample_interval=SAMPLE_INTERVAL,
        target_column="load",
        forecast_start=with_covariate.forecast_start,
    )

    # Act
    covariate_forecast = forecaster.predict(with_covariate)
    baseline_forecast = forecaster.predict(without_covariate)

    # Assert
    assert covariate_forecast.quantiles_data.shape == baseline_forecast.quantiles_data.shape
    assert np.isfinite(covariate_forecast.quantiles_data.to_numpy()).all()
    # Conditioning on the covariate changes the model's output.
    assert not np.allclose(
        covariate_forecast.quantiles_data.to_numpy(),
        baseline_forecast.quantiles_data.to_numpy(),
    )
