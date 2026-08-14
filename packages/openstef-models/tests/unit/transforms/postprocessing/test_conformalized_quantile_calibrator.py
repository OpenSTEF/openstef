# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from openstef_core.datasets import ForecastDataset
from openstef_core.types import Quantile
from openstef_models.transforms.postprocessing import ConformalizedQuantileCalibrator


def _dataset(predictions: np.ndarray, actuals: np.ndarray) -> ForecastDataset:
    """Build a forecast dataset with P10, P50, and P90 columns."""
    return ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P10": predictions[:, 0],
                "quantile_P50": predictions[:, 1],
                "quantile_P90": predictions[:, 2],
                "load": actuals,
            },
            index=pd.date_range("2025-01-01", periods=len(actuals), freq="h"),
        )
    )


def test_conformalized_calibrator_applies_reference_tail_corrections() -> None:
    """Test asymmetric lower and upper corrections while preserving the median."""
    predictions = np.column_stack([np.zeros(100), np.full(100, 5.0), np.full(100, 10.0)])
    actuals = np.concatenate([np.full(90, 1.0), np.full(10, 20.0)])
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(predictions[:2], np.full(2, 5.0))

    calibrator = ConformalizedQuantileCalibrator(quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)])
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    np.testing.assert_allclose(result.data["quantile_P10"], [1.0, 1.0])
    np.testing.assert_allclose(result.data["quantile_P50"], [5.0, 5.0])
    np.testing.assert_allclose(result.data["quantile_P90"], [20.0, 20.0])


def test_conformalized_calibrator_can_calibrate_the_median_when_enabled() -> None:
    """Test that median calibration is explicitly opt-in."""
    predictions = np.column_stack([np.zeros(100), np.zeros(100), np.full(100, 10.0)])
    actuals = np.full(100, 5.0)
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(predictions[:1], np.full(1, 5.0))

    calibrator = ConformalizedQuantileCalibrator(
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        conformalize_median=True,
    )
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    np.testing.assert_allclose(result.data["quantile_P50"], [5.0])


def test_conformalized_calibrator_skips_short_calibration_windows() -> None:
    """Test that too few calibration samples leave forecasts unchanged."""
    predictions = np.column_stack([np.zeros(2), np.full(2, 5.0), np.full(2, 10.0)])
    actuals = np.full(2, 20.0)
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(predictions[:1], np.full(1, 5.0))

    calibrator = ConformalizedQuantileCalibrator(
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        min_calibration_samples=3,
    )
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    np.testing.assert_allclose(result.data["quantile_P10"], [0.0])
    np.testing.assert_allclose(result.data["quantile_P90"], [10.0])


def test_conformalized_calibrator_skips_only_sparse_quantiles() -> None:
    """Test that sparse quantiles are skipped while sufficiently populated ones calibrate."""
    predictions = np.column_stack(
        [
            np.concatenate([np.zeros(100), [np.nan]]),
            np.full(101, 5.0),
            np.full(101, 10.0),
        ]
    )
    actuals = np.full(101, 20.0)
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(np.array([[0.0, 5.0, 10.0]]), np.array([5.0]))

    calibrator = ConformalizedQuantileCalibrator(
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        min_calibration_samples=101,
    )
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    np.testing.assert_allclose(result.data["quantile_P10"], [0.0])
    np.testing.assert_allclose(result.data["quantile_P90"], [20.0])


def test_conformalized_calibrator_does_not_sort_quantiles() -> None:
    """Test that ordering remains the responsibility of a downstream sorter."""
    predictions = np.column_stack([np.full(2, 10.0), np.zeros(2), np.full(2, 5.0)])
    actuals = np.full(2, 5.0)
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(predictions[:1], np.full(1, 5.0))

    calibrator = ConformalizedQuantileCalibrator(
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        min_calibration_samples=1,
    )
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    np.testing.assert_allclose(result.data.filter(like="quantile_").to_numpy(), [[5.0, 0.0, 5.0]])
