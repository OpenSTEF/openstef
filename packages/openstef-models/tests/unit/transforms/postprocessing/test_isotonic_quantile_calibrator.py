# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd

from openstef_core.datasets import ForecastDataset
from openstef_core.types import Quantile
from openstef_models.transforms.postprocessing.isotonic_quantile_calibrator import (
    IsotonicQuantileCalibrator,
)


def create_heterogeneous_validation_data(n_samples: int = 500, seed: int = 42) -> ForecastDataset:
    """Create validation data where P90 bias varies by prediction magnitude.

    Creates three regimes with different systematic biases to test global vs local windowed fit:
    - Low regime: Model underestimates (15% exceedance instead of 10%)
    - Medium regime: Well-calibrated (10% exceedance)
    - High regime: Model overestimates (5% exceedance instead of 10%)
    """
    rng = np.random.default_rng(seed)

    # Create three regimes with different prediction magnitudes
    predictions_low = rng.uniform(10, 50, n_samples // 3)
    predictions_med = rng.uniform(50, 100, n_samples // 3)
    predictions_high = rng.uniform(100, 150, n_samples // 3)

    # Create actuals with different bias patterns
    actuals_low = _create_biased_regime(predictions_low, exceedance_rate=0.15, center_offset=-5, noise_std=8, rng=rng)
    actuals_med = _create_biased_regime(predictions_med, exceedance_rate=0.10, center_offset=-3, noise_std=5, rng=rng)
    actuals_high = _create_biased_regime(
        predictions_high, exceedance_rate=0.05, center_offset=-10, noise_std=10, rng=rng
    )

    # Combine all regimes
    predictions_p90 = np.concatenate([predictions_low, predictions_med, predictions_high])
    actuals = np.concatenate([actuals_low, actuals_med, actuals_high])

    # Shuffle to mix regimes
    indices = rng.permutation(len(predictions_p90))
    predictions_p90 = predictions_p90[indices]
    actuals = actuals[indices]

    return _create_forecast_dataset(predictions_p90, actuals, quantile_level=0.9)


def _create_biased_regime(
    predictions: np.ndarray, exceedance_rate: float, center_offset: float, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    """Create actuals with systematic bias relative to predictions."""
    actuals = rng.normal(predictions + center_offset, noise_std)
    # Create exceedances at the specified rate
    return np.where(
        rng.random(len(actuals)) > (1 - exceedance_rate),
        predictions + rng.exponential(noise_std * 0.6, len(actuals)),
        actuals,
    )


def _create_forecast_dataset(
    predictions: np.ndarray, actuals: np.ndarray, quantile_level: float = 0.9, start_date: str = "2024-01-01"
) -> ForecastDataset:
    """Create a ForecastDataset from predictions and actuals."""
    index = pd.date_range(start_date, periods=len(predictions), freq="h")
    quantile_col = Quantile(quantile_level).format()
    data = pd.DataFrame({quantile_col: predictions, "load": actuals}, index=index)
    return ForecastDataset(data=data)


def _split_train_test(data: ForecastDataset, train_ratio: float = 0.7) -> tuple[ForecastDataset, ForecastDataset]:
    """Split dataset into train and test sets."""
    train_size = int(len(data.data) * train_ratio)
    train = ForecastDataset(data=data.data.iloc[:train_size])
    test = ForecastDataset(data=data.data.iloc[train_size:])
    return train, test


def _compute_coverage(target_series: pd.Series | None, predictions: pd.Series) -> float:
    """Compute percentage of actuals that fall below predictions."""
    return float(np.mean(target_series <= predictions) * 100)


def _verify_calibration_results(calibrated_global: ForecastDataset, calibrated_local: ForecastDataset):
    """Verify that calibration results are valid."""
    assert calibrated_global.data["quantile_P90"].notna().all()
    assert calibrated_local.data["quantile_P90"].notna().all()


def test_isotonic_calibrator_local_vs_global():
    """Demonstrate the difference between local quantile estimation and direct isotonic regression."""
    validation_data = create_heterogeneous_validation_data()
    train_data, test_data = _split_train_test(validation_data)
    target_quantile = 90.0

    # Method 1: Direct isotonic regression (global fit)
    calibrator_global = IsotonicQuantileCalibrator(
        quantiles=[Quantile(target_quantile / 100)], use_local_quantile_estimation=False
    )
    calibrator_global.fit(train_data)
    calibrated_global = calibrator_global.transform(test_data)

    # Method 2: Local quantile estimation + isotonic regression
    calibrator_local = IsotonicQuantileCalibrator(
        quantiles=[Quantile(target_quantile / 100)], use_local_quantile_estimation=True, window_size=50
    )
    calibrator_local.fit(train_data)
    calibrated_local = calibrator_local.transform(test_data)

    # Compute coverage metrics
    coverage_test_data = _compute_coverage(test_data.target_series, test_data.data["quantile_P90"])
    coverage_global_calibrator = _compute_coverage(test_data.target_series, calibrated_global.data["quantile_P90"])
    coverage_local_calibrator = _compute_coverage(test_data.target_series, calibrated_local.data["quantile_P90"])

    _verify_calibration_results(calibrated_global, calibrated_local)

    # Expectation: calibration should bring coverage closer to the target quantile (90%)
    assert abs(coverage_local_calibrator - target_quantile) < abs(coverage_test_data - target_quantile)

    # Expectation: local calibration should be better than global because bias is heterogeneous in the data
    assert abs(coverage_local_calibrator - target_quantile) < abs(coverage_global_calibrator - target_quantile)


def test_isotonic_calibrator_nonlinear_data():
    """Demonstrate that isotonic regression can fit non-linear monotonic relationships."""
    # predictions with quadratic underestimation
    train_predictions = np.arange(10.0, 101.0, 10.0)
    # true P90 higher, increasing quadratically
    train_actuals = train_predictions + (train_predictions / 20) ** 2

    train_data = _create_forecast_dataset(train_predictions, train_actuals, quantile_level=0.9, start_date="2025-01-01")

    # new test predictions with same pattern
    test_predictions = np.array([25.0, 45, 65, 85])
    test_actuals = test_predictions + (test_predictions / 20) ** 2
    test_data = _create_forecast_dataset(test_predictions, test_actuals, quantile_level=0.9, start_date="2025-02-01")

    # 0 percent coverage before calibration
    coverage_before = _compute_coverage(test_data.target_series, test_data.data["quantile_P90"])
    assert coverage_before == 0.0

    # fit calibrator on training data
    calibrator = IsotonicQuantileCalibrator(quantiles=[Quantile(0.9)])
    calibrator.fit(train_data)
    calibrated = calibrator.transform(test_data)

    # coverage after calibration
    coverage_after = _compute_coverage(test_data.target_series, calibrated.data["quantile_P90"])

    # Expectation: coverage closer to P90 after calibration
    assert abs(coverage_after - 90.0) < abs(coverage_before - 90.0)


def test_isotonic_calibrator_does_not_capture_non_monotonic():
    """Demonstrate that isotonic regression cannot capture non-monotonic (U-shaped) relationships.

    Isotonic regression can only fit monotonic functions. When the true
    relationship is non-monotonic (e.g., U-shaped), isotonic regression still returns a monotonic fit.
    """
    # non-monotonic bias: actuals follow U-shape (decreasing with increase in prediction and increasing half way)
    # while prediction increase linearly
    train_predictions = np.linspace(10.0, 100.0, num=10)
    # U-shape: quadratic with minimum at center
    center = (train_predictions.max() + train_predictions.min()) / 2
    train_actuals = 20 + ((train_predictions - center) / 10) ** 2 * 20

    # the diff between one value and the next is not always positive in actuals
    assert not np.all(np.diff(train_actuals) >= 0)
    # while it is always positive in predictions
    assert np.all(np.diff(train_predictions) >= 0)

    train_data = _create_forecast_dataset(train_predictions, train_actuals, quantile_level=0.9, start_date="2025-01-01")

    # Fit isotonic calibrator
    calibrator = IsotonicQuantileCalibrator(quantiles=[Quantile(0.9)])
    calibrator.fit(train_data)

    # Apply to test data
    test_predictions = np.array([15.0, 35.0, 55.0, 85.0])
    test_actuals = np.array([90.0, 50.0, 30.0, 70.0])  # U-shape
    test_data = _create_forecast_dataset(test_predictions, test_actuals, quantile_level=0.9, start_date="2025-02-01")
    calibrated = calibrator.transform(test_data)

    coverage_before = _compute_coverage(test_data.target_series, test_data.data["quantile_P90"])
    coverage_after = _compute_coverage(test_data.target_series, calibrated.data["quantile_P90"])

    # Key observation: isotonic regression enforces monotonicity
    calibrated_predictions = calibrated.data["quantile_P90"].to_numpy()  # pyright: ignore[reportUnknownMemberType]
    assert np.all(np.diff(calibrated_predictions) >= 0)

    # Coverage not more than 75% improvement after calibration
    improvement_calibration = abs(coverage_before - 90.0) - abs(coverage_after - 90.0)
    assert improvement_calibration <= abs(coverage_before - 90.0) * 0.75
