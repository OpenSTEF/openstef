# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import Quantile
from openstef_models.transforms.postprocessing import MapieQuantileCalibrator


def _dataset(predictions: np.ndarray, actuals: np.ndarray) -> ForecastDataset:
    quantile_levels = [0.1, 0.5, 0.9] if predictions.shape[1] == 3 else [0.1, 0.3, 0.5, 0.7, 0.9]
    return ForecastDataset(
        data=pd.DataFrame(
            {
                **{Quantile(level).format(): predictions[:, index] for index, level in enumerate(quantile_levels)},
                "load": actuals,
            },
            index=pd.date_range("2025-01-01", periods=len(actuals), freq="h"),
        )
    )


def test_mapie_calibrator_conformalizes_outer_quantiles() -> None:
    predictions = np.column_stack([np.zeros(100), np.full(100, 5.0), np.full(100, 10.0)])
    actuals = np.concatenate([np.full(90, 5.0), np.full(10, 20.0)])
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(predictions[:2], np.full(2, 5.0))

    calibrator = MapieQuantileCalibrator(quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)])
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    assert calibrator.is_fitted
    np.testing.assert_allclose(result.data["quantile_P10"], [5.0, 5.0])
    np.testing.assert_allclose(result.data["quantile_P50"], [5.0, 5.0])
    assert np.all(result.data["quantile_P90"] > 10.0)


def test_mapie_calibrator_requires_fit_before_transform() -> None:
    calibrator = MapieQuantileCalibrator()

    with pytest.raises(NotFittedError):
        calibrator.transform(_dataset(np.ones((1, 3)), np.ones(1)))


def test_mapie_calibrator_rejects_duplicate_quantiles() -> None:
    calibrator = MapieQuantileCalibrator(quantiles=[Quantile(0.1), Quantile(0.1)])

    with pytest.raises(ValueError, match="unique quantiles"):
        calibrator.fit(_dataset(np.ones((1, 3)), np.ones(1)))


def test_mapie_calibrator_calibrates_each_requested_quantile_independently() -> None:
    predictions = np.column_stack(
        [np.zeros(100), np.full(100, 5.0), np.full(100, 10.0), np.full(100, 15.0), np.full(100, 20.0)]
    )
    actuals = np.concatenate([np.full(90, 10.0), np.full(10, 30.0)])
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(predictions[:2], np.full(2, 10.0))

    calibrator = MapieQuantileCalibrator(quantiles=[Quantile(level) for level in [0.1, 0.3, 0.5, 0.7, 0.9]])
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    assert list(result.data.filter(like="quantile_").columns) == [
        "quantile_P10",
        "quantile_P30",
        "quantile_P50",
        "quantile_P70",
        "quantile_P90",
    ]
    assert np.all(result.data["quantile_P10"] > predictions[0, 0])
    assert np.all(np.isfinite(result.data["quantile_P50"]))
    assert np.all(result.data["quantile_P90"] > predictions[0, 4])


def test_mapie_calibrator_restores_quantile_ordering_after_independent_corrections() -> None:
    predictions = np.tile([0.0, 10.0, 20.0], (100, 1))
    actuals = np.concatenate([np.full(90, 30.0), np.full(10, 0.0)])
    calibration = _dataset(predictions, actuals)
    forecast = _dataset(predictions[:2], np.full(2, 10.0))

    calibrator = MapieQuantileCalibrator(quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)])
    calibrator.fit(calibration)
    result = calibrator.transform(forecast)

    quantile_values = result.data.filter(like="quantile_").to_numpy()
    assert np.all(np.diff(quantile_values, axis=1) >= 0)
