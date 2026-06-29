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

from openstef_core.datasets.validated_datasets import ForecastDataset, ForecastInputDataset
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.models.checkpoint import CheckpointMetadata
from openstef_foundation_models.models.forecasting import Chronos2Forecaster

# Chronos-2's native quantile grid (21 levels).
NATIVE_QUANTILES = [
    Quantile(0.01), Quantile(0.05), Quantile(0.1), Quantile(0.15), Quantile(0.2),
    Quantile(0.25), Quantile(0.3), Quantile(0.35), Quantile(0.4), Quantile(0.45),
    Quantile(0.5), Quantile(0.55), Quantile(0.6), Quantile(0.65), Quantile(0.7),
    Quantile(0.75), Quantile(0.8), Quantile(0.85), Quantile(0.9), Quantile(0.95),
    Quantile(0.99),
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
        input_names=["context", "group_ids", "attention_mask", "future_covariates", "future_covariates_mask"],
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


def _make_input_with_covariates(
    covariates: list[str],
    periods: int = 100,
    forecast_offset: int = 80,
) -> ForecastInputDataset:
    """Build a forecast input with a ramp target plus known covariate columns.

    Each covariate spans the full index (history and horizon). Covariate ``c``
    at step ``t`` holds ``t + 1000 * (c_index + 1)`` so values are distinct per
    covariate and easy to verify.

    Args:
        covariates: Names of the covariate columns to add.
        periods: Total number of timesteps in the series.
        forecast_offset: Index of the forecast start within the series.
    """
    index = pd.date_range("2025-01-01", periods=periods, freq=SAMPLE_INTERVAL)
    frame = pd.DataFrame({"load": np.arange(periods, dtype=float)}, index=index)
    for offset, name in enumerate(covariates, start=1):
        frame[name] = np.arange(periods, dtype=float) + 1000.0 * offset
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
    first_result, second_result = results
    assert isinstance(first_result, ForecastDataset)
    assert isinstance(second_result, ForecastDataset)
    assert first_result.data.index[0].to_pydatetime() == first.forecast_start
    assert second_result.data.index[0].to_pydatetime() == second.forecast_start


def test_supports_batching_is_enabled(forecaster: Chronos2Forecaster) -> None:
    """Chronos-2 advertises batch support."""
    # Assert
    assert forecaster.supports_batching is True


def test_build_inputs_adds_a_covariate_row_per_feature_sharing_group_id(forecaster: Chronos2Forecaster) -> None:
    """A series with K covariates produces 1 + K rows that share one group id."""
    # Arrange
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)
    data = _make_input_with_covariates(["temperature", "radiation"])

    # Act
    forecaster.predict(data)

    # Assert: target row + 2 covariate rows, all in group 0
    assert backend.last_inputs is not None
    assert backend.last_inputs["context"].shape == (3, CONTEXT_LENGTH)
    np.testing.assert_array_equal(backend.last_inputs["group_ids"], np.array([0, 0, 0], dtype=np.int64))


def test_build_inputs_covariate_context_holds_covariate_history(forecaster: Chronos2Forecaster) -> None:
    """Each covariate row's context carries that covariate's own history."""
    # Arrange
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)
    data = _make_input_with_covariates(["temperature"], periods=100, forecast_offset=80)

    # Act
    forecaster.predict(data)

    # Assert: covariate row (index 1) context ends at the value just before
    # forecast start. Covariate at step t = t + 1000; step 79 -> 1079.
    assert backend.last_inputs is not None
    covariate_context = backend.last_inputs["context"][1]
    assert covariate_context[-1] == pytest.approx(1079.0)


def test_build_inputs_future_covariates_carry_known_horizon_values(forecaster: Chronos2Forecaster) -> None:
    """Covariate rows carry known horizon values; the target row is masked out."""
    # Arrange: 120 steps so the full 32-step horizon after index 80 is covered.
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)
    data = _make_input_with_covariates(["temperature"], periods=120, forecast_offset=80)

    # Act
    forecaster.predict(data)

    # Assert
    assert backend.last_inputs is not None
    future = backend.last_inputs["future_covariates"]
    future_mask = backend.last_inputs["future_covariates_mask"]
    horizon = backend.metadata.horizon_length

    # Target row (0): future fully masked out.
    np.testing.assert_array_equal(future_mask[0], np.zeros(horizon, dtype=np.float32))
    np.testing.assert_array_equal(future[0], np.zeros(horizon, dtype=np.float32))

    # Covariate row (1): forecast start is index 80, covariate step t = t + 1000.
    # The horizon spans steps 80..80+horizon-1, all known.
    np.testing.assert_array_equal(future_mask[1], np.ones(horizon, dtype=np.float32))
    expected = np.arange(80, 80 + horizon, dtype=np.float32) + 1000.0
    np.testing.assert_array_almost_equal(future[1], expected)


def test_build_inputs_future_covariates_masked_beyond_available_horizon(forecaster: Chronos2Forecaster) -> None:
    """Horizon steps with no covariate value are zero-filled and masked out."""
    # Arrange: only 4 steps of covariate exist past the forecast start, but the
    # model horizon is 32 steps.
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)
    data = _make_input_with_covariates(["temperature"], periods=84, forecast_offset=80)

    # Act
    forecaster.predict(data)

    # Assert
    assert backend.last_inputs is not None
    future_mask = backend.last_inputs["future_covariates_mask"][1]
    horizon = backend.metadata.horizon_length
    # Steps 80..83 (4 values) known, the remaining horizon masked out.
    assert future_mask[:4].sum() == pytest.approx(4.0)
    np.testing.assert_array_equal(future_mask[4:], np.zeros(horizon - 4, dtype=np.float32))


def test_predict_batch_assigns_one_group_per_series_with_covariates(forecaster: Chronos2Forecaster) -> None:
    """Batched series with covariates get contiguous, per-series group ids."""
    # Arrange
    backend = forecaster.backend
    assert isinstance(backend, RecordingBackend)
    batch = [
        _make_input_with_covariates(["temperature", "radiation"]),
        _make_input_with_covariates(["temperature"]),
    ]

    # Act
    forecaster.predict_batch(batch)

    # Assert: series 0 -> 3 rows (group 0), series 1 -> 2 rows (group 1)
    assert backend.last_inputs is not None
    assert backend.last_inputs["context"].shape == (5, CONTEXT_LENGTH)
    np.testing.assert_array_equal(backend.last_inputs["group_ids"], np.array([0, 0, 0, 1, 1], dtype=np.int64))


def test_predict_slices_the_target_row_from_grouped_output() -> None:
    """With covariates present, post-processing reads each series' target row."""
    # Arrange: a backend whose output encodes each row's global index as the
    # constant median, so we can verify which row is read back.
    metadata = CheckpointMetadata(
        model_family="chronos2",
        input_names=["context", "group_ids", "attention_mask", "future_covariates", "future_covariates_mask"],
        output_name="quantile_preds",
        native_quantiles=NATIVE_QUANTILES,
        context_length=CONTEXT_LENGTH,
        output_patch_size=OUTPUT_PATCH_SIZE,
        horizon_patches=HORIZON_PATCHES,
        resolution_minutes=15,
    )
    backend = RowIndexBackend(metadata)
    forecaster = Chronos2Forecaster(
        backend=backend,
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime.from_string("PT2H")],
    )
    batch = [
        _make_input_with_covariates(["temperature", "radiation"]),  # rows 0,1,2 -> target row 0
        _make_input_with_covariates(["temperature"]),  # rows 3,4 -> target row 3
    ]

    # Act
    results = forecaster.predict_batch(batch)

    # Assert: each forecast median equals its target row's global index.
    first_result, second_result = results
    assert isinstance(first_result, ForecastDataset)
    assert isinstance(second_result, ForecastDataset)
    median = Quantile(0.5).format()
    assert first_result.data[median].iloc[0] == pytest.approx(0.0)
    assert second_result.data[median].iloc[0] == pytest.approx(3.0)


class RowIndexBackend:
    """A stub backend whose output encodes each row's global index.

    Every native quantile of row ``r`` is the constant value ``r``. Slicing a
    target row therefore yields a forecast whose value is that row's index,
    which makes target-row selection observable.
    """

    def __init__(self, metadata: CheckpointMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> CheckpointMetadata:
        return self._metadata

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        batch_size = inputs["context"].shape[0]
        horizon = self._metadata.horizon_length
        n_quantiles = len(NATIVE_QUANTILES)
        row_index = np.arange(batch_size, dtype=np.float32)
        output = np.broadcast_to(row_index[:, None, None], (batch_size, n_quantiles, horizon))
        return {self._metadata.output_name: output.copy()}

    def close(self) -> None:
        pass


def _frozen_covariate_forecaster(max_covariates: int) -> tuple[Chronos2Forecaster, RecordingBackend]:
    """Build a forecaster whose checkpoint froze its covariate axis at *max_covariates*."""
    metadata = CheckpointMetadata(
        model_family="chronos2",
        input_names=["context", "group_ids", "attention_mask", "future_covariates", "future_covariates_mask"],
        output_name="quantile_preds",
        native_quantiles=NATIVE_QUANTILES,
        context_length=CONTEXT_LENGTH,
        output_patch_size=OUTPUT_PATCH_SIZE,
        horizon_patches=HORIZON_PATCHES,
        resolution_minutes=15,
        max_covariates=max_covariates,
    )
    backend = RecordingBackend(metadata)
    forecaster = Chronos2Forecaster(
        backend=backend,
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime.from_string("PT2H")],
    )
    return forecaster, backend


def test_build_inputs_pads_short_series_to_frozen_covariate_axis() -> None:
    """A frozen covariate axis pads a short series up to a fixed row count with masked rows."""
    # Arrange: axis frozen at 3 covariates, but the series carries only 1.
    forecaster, backend = _frozen_covariate_forecaster(max_covariates=3)
    data = _make_input_with_covariates(["temperature"], periods=120, forecast_offset=80)

    # Act
    forecaster.predict(data)

    # Assert: target + 3 covariate rows (1 real, 2 padded), all in group 0.
    assert backend.last_inputs is not None
    assert backend.last_inputs["context"].shape == (4, CONTEXT_LENGTH)
    np.testing.assert_array_equal(backend.last_inputs["group_ids"], np.zeros(4, dtype=np.int64))
    # The 2 padded rows (indices 2,3) are fully masked and zero-filled, so the
    # model ignores them across both context and horizon.
    horizon = backend.metadata.horizon_length
    np.testing.assert_array_equal(backend.last_inputs["attention_mask"][2:], np.zeros((2, CONTEXT_LENGTH), np.float32))
    np.testing.assert_array_equal(backend.last_inputs["context"][2:], np.zeros((2, CONTEXT_LENGTH), np.float32))
    np.testing.assert_array_equal(backend.last_inputs["future_covariates_mask"][2:], np.zeros((2, horizon), np.float32))


def test_build_inputs_accepts_covariates_matching_frozen_axis() -> None:
    """A series whose covariate count matches the frozen axis is passed through unpadded."""
    # Arrange
    forecaster, backend = _frozen_covariate_forecaster(max_covariates=2)
    data = _make_input_with_covariates(["temperature", "radiation"])

    # Act
    forecaster.predict(data)

    # Assert: target + exactly 2 covariate rows, no padding.
    assert backend.last_inputs is not None
    assert backend.last_inputs["context"].shape == (3, CONTEXT_LENGTH)


def test_build_inputs_rejects_more_covariates_than_frozen_axis() -> None:
    """A series with more covariates than the frozen axis fails with a clear message, not an ORT error."""
    # Arrange: axis frozen at 1 covariate, series carries 2.
    forecaster, _ = _frozen_covariate_forecaster(max_covariates=1)
    data = _make_input_with_covariates(["temperature", "radiation"])

    # Act / Assert
    with pytest.raises(ValueError, match="covariate axis is frozen at 1"):
        forecaster.predict(data)
