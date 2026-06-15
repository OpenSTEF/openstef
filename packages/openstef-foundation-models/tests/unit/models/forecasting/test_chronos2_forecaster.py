# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for :class:`Chronos2Forecaster`.

The inference backend is replaced with a small recording stub so the tests
exercise the forecaster's pre- and post-processing without ONNX Runtime or any
checkpoint artifact.
"""

from collections.abc import Mapping
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.models.checkpoint import CheckpointMetadata
from openstef_foundation_models.models.forecasting import Chronos2Forecaster

# Chronos-2's native quantile grid (21 levels).
NATIVE_QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]  # fmt: skip

CONTEXT_LENGTH = 64
OUTPUT_PATCH_SIZE = 16
HORIZON_PATCHES = 2  # horizon_length = 32 steps
SAMPLE_INTERVAL = timedelta(minutes=15)


class RecordingBackend:
    """A stub :class:`InferenceBackend` that records inputs and returns a ramp.

    The output for each series is a quantile ramp: native level ``i`` maps to a
    constant value of ``i * 10`` across the whole horizon. This makes the
    post-processed quantiles easy to predict by hand.
    """

    def __init__(self, metadata: CheckpointMetadata) -> None:
        self._metadata = metadata
        self.last_inputs: Mapping[str, np.ndarray] | None = None

    @property
    def metadata(self) -> CheckpointMetadata:
        return self._metadata

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        self.last_inputs = {key: np.array(value) for key, value in inputs.items()}
        batch_size = inputs["context"].shape[0]
        horizon = self._metadata.horizon_length
        ramp = np.arange(len(NATIVE_QUANTILES), dtype=np.float32) * 10.0
        per_series = np.broadcast_to(ramp[:, None], (len(NATIVE_QUANTILES), horizon))
        return {self._metadata.output_name: np.broadcast_to(per_series, (batch_size, *per_series.shape)).copy()}

    def close(self) -> None:
        pass


@pytest.fixture
def metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        model_family="chronos2",
        input_names=["context", "group_ids", "attention_mask"],
        output_name="quantile_preds",
        native_quantiles=NATIVE_QUANTILES,
        context_length=CONTEXT_LENGTH,
        output_patch_size=OUTPUT_PATCH_SIZE,
        horizon_patches=HORIZON_PATCHES,
        resolution_minutes=15,
    )


@pytest.fixture
def backend(metadata: CheckpointMetadata) -> RecordingBackend:
    return RecordingBackend(metadata)


@pytest.fixture
def forecaster(backend: RecordingBackend) -> Chronos2Forecaster:
    return Chronos2Forecaster(
        backend=backend,
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime.from_string("PT2H")],
    )


def _make_input(periods: int = 100, forecast_offset: int = 80) -> ForecastInputDataset:
    """Build a forecast input with a simple ramp target series.

    Args:
        periods: Total number of timesteps in the series.
        forecast_offset: Index of the forecast start within the series.
    """
    index = pd.date_range("2025-01-01", periods=periods, freq=SAMPLE_INTERVAL)
    frame = pd.DataFrame({"load": np.arange(periods, dtype=float)}, index=index)
    return ForecastInputDataset(
        data=frame,
        sample_interval=SAMPLE_INTERVAL,
        target_column="load",
        forecast_start=index[forecast_offset].to_pydatetime(),
    )


def test_chronos2_forecaster_is_always_fitted(forecaster: Chronos2Forecaster) -> None:
    """The zero-shot model needs no training, so it is fitted on construction."""
    # Assert
    assert forecaster.is_fitted is True


def test_chronos2_forecaster_fit_is_a_noop(forecaster: Chronos2Forecaster) -> None:
    """Calling fit does not raise and leaves the model fitted."""
    # Act
    forecaster.fit(_make_input())

    # Assert
    assert forecaster.is_fitted is True


def test_predict_returns_requested_quantile_columns(forecaster: Chronos2Forecaster) -> None:
    """The forecast carries exactly the requested quantile columns."""
    # Act
    result = forecaster.predict(_make_input())

    # Assert
    expected = [Quantile(0.1).format(), Quantile(0.5).format(), Quantile(0.9).format()]
    assert list(result.data.columns) == expected


def test_predict_index_starts_at_forecast_start(forecaster: Chronos2Forecaster) -> None:
    """The forecast index begins at the input's forecast start."""
    # Arrange
    data = _make_input()

    # Act
    result = forecaster.predict(data)

    # Assert
    assert result.data.index[0].to_pydatetime() == data.forecast_start


def test_predict_horizon_is_capped_to_requested_lead_time(forecaster: Chronos2Forecaster) -> None:
    """A PT2H horizon at 15-minute resolution yields 9 inclusive steps."""
    # Act
    result = forecaster.predict(_make_input())

    # Assert: range [start, start + 2h] inclusive at 15 min => 9 points
    assert len(result.data) == 9


def test_predict_horizon_is_capped_to_model_horizon_length(backend: RecordingBackend) -> None:
    """A requested horizon longer than the model's frozen horizon is clipped."""
    # Arrange: model emits 32 steps; ask for PT24H (97 steps) at 15 min
    forecaster = Chronos2Forecaster(
        backend=backend,
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime.from_string("PT24H")],
    )

    # Act
    result = forecaster.predict(_make_input(periods=200, forecast_offset=100))

    # Assert: clipped to the model's horizon_length
    assert len(result.data) == backend.metadata.horizon_length


def test_predict_resamples_native_quantiles_onto_requested_grid(forecaster: Chronos2Forecaster) -> None:
    """Requested quantiles are interpolated from the native ramp output."""
    # Act
    result = forecaster.predict(_make_input())

    # Assert: native level i -> value i*10. P10 is index 2 (0.1) -> 20,
    # P50 is index 10 (0.5) -> 100, P90 is index 18 (0.9) -> 180.
    np.testing.assert_array_almost_equal(result.data[Quantile(0.1).format()], np.full(9, 20.0))
    np.testing.assert_array_almost_equal(result.data[Quantile(0.5).format()], np.full(9, 100.0))
    np.testing.assert_array_almost_equal(result.data[Quantile(0.9).format()], np.full(9, 180.0))


def test_build_inputs_uses_recent_context_with_full_mask(forecaster: Chronos2Forecaster) -> None:
    """The context is the most recent values before forecast start, fully observed."""
    # Arrange
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)

    # Act
    forecaster.predict(_make_input(periods=100, forecast_offset=80))

    # Assert
    assert backend.last_inputs is not None
    context = backend.last_inputs["context"]
    mask = backend.last_inputs["attention_mask"]
    assert context.shape == (1, CONTEXT_LENGTH)
    # Forecast start is index 80; the last context value is the target at index 79.
    assert context[0, -1] == pytest.approx(79.0)
    # All context values are observed.
    np.testing.assert_array_equal(mask, np.ones((1, CONTEXT_LENGTH), dtype=np.float32))


def test_build_inputs_left_pads_and_masks_short_history(forecaster: Chronos2Forecaster) -> None:
    """A history shorter than the context window is left-padded and masked out."""
    # Arrange: only 10 values before the forecast start
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)
    data = _make_input(periods=20, forecast_offset=10)

    # Act
    forecaster.predict(data)

    # Assert
    assert backend.last_inputs is not None
    mask = backend.last_inputs["attention_mask"][0]
    context = backend.last_inputs["context"][0]
    # 10 observed values at the end, the rest padded.
    assert mask.sum() == pytest.approx(10.0)
    np.testing.assert_array_equal(mask[-10:], np.ones(10, dtype=np.float32))
    np.testing.assert_array_equal(mask[:-10], np.zeros(CONTEXT_LENGTH - 10, dtype=np.float32))
    # Padded positions are zero-filled.
    np.testing.assert_array_equal(context[:-10], np.zeros(CONTEXT_LENGTH - 10, dtype=np.float32))


def test_predict_batch_runs_backend_once_for_all_series(forecaster: Chronos2Forecaster) -> None:
    """A batch is forecast in a single backend call with sequential group ids."""
    # Arrange
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)
    batch = [_make_input(), _make_input(forecast_offset=70)]

    # Act
    results = forecaster.predict_batch(batch)

    # Assert
    assert len(results) == 2
    assert backend.last_inputs is not None
    assert backend.last_inputs["context"].shape == (2, CONTEXT_LENGTH)
    np.testing.assert_array_equal(backend.last_inputs["group_ids"], np.array([0, 1], dtype=np.int64))


def test_predict_batch_preserves_per_series_forecast_start(forecaster: Chronos2Forecaster) -> None:
    """Each batched forecast keeps its own input's forecast start."""
    # Arrange
    first = _make_input(forecast_offset=80)
    second = _make_input(forecast_offset=70)

    # Act
    results = forecaster.predict_batch([first, second])

    # Assert
    assert results[0].data.index[0].to_pydatetime() == first.forecast_start
    assert results[1].data.index[0].to_pydatetime() == second.forecast_start


def test_supports_batching_is_enabled(forecaster: Chronos2Forecaster) -> None:
    """Chronos-2 advertises batch support."""
    # Assert
    assert forecaster.supports_batching is True
