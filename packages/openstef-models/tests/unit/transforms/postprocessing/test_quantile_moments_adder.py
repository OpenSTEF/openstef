# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastDataset
from openstef_core.mixins import TransformPipeline
from openstef_core.types import Quantile
from openstef_models.transforms.postprocessing import QuantileMomentsAdder, QuantileSorter


@pytest.mark.parametrize(
    ("levels", "values", "mean", "stdev"),
    [
        # Trapezoidal averages of the values and squared deviations, normalized by the P05 to P95 span.
        pytest.param(
            [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            [70, 78, 88, 95, 110, 128, 140, 158, 175],
            1361 / 12,
            np.sqrt(96875 / 144),
            id="nine-quantiles",
        ),
        pytest.param(
            [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            [70, 78, 95, 110, 128, 158, 175],
            1363 / 12,
            np.sqrt(11731 / 16),
            id="seven-quantiles",
        ),
        pytest.param([0.9, 0.1, 0.5], [120, 90, 100], 102.5, np.sqrt(118.75), id="unsorted-probability-levels"),
        pytest.param([0.025, 0.997], [-4, 8], 2, 6, id="two-quantiles"),
        pytest.param([0.1, 0.7, 0.9], [0.1, 0.1, 0.1], 0.1, 0, id="constant-forecast"),
    ],
)
def test_moments_follow_the_available_grid(levels: list[float], values: list[float], mean: float, stdev: float) -> None:
    quantiles = [Quantile(level) for level in levels]
    index = pd.date_range("2026-01-01", periods=1, tz="UTC", freq="15min")
    predictions = pd.DataFrame([values], columns=[q.format() for q in quantiles], index=index)
    predictions["std"] = 999
    original = ForecastDataset(predictions, standard_deviation_column="std")

    result = QuantileMomentsAdder().transform(original)
    round_trip = ForecastDataset(result.to_pandas())

    assert round_trip.data["forecast_mean"].iloc[0] == pytest.approx(mean)
    assert round_trip.standard_deviation_series.iloc[0] == pytest.approx(stdev)
    assert round_trip.quantiles == quantiles
    pd.testing.assert_frame_equal(round_trip.quantiles_data, original.quantiles_data)


def test_missing_values_keep_their_probability_levels_when_sorting() -> None:
    predictions = ForecastDataset(
        pd.DataFrame(
            {
                "quantile_P10": [np.nan, np.nan, 90.0],
                "quantile_P30": [100.0, 100.0, 100.0],
                "quantile_P50": [140.0, np.inf, 120.0],
                "quantile_P90": [120.0, np.nan, 110.0],
            },
            index=pd.date_range("2026-01-01", periods=3, tz="UTC", freq="15min"),
        )
    )
    transform = QuantileMomentsAdder()
    pipeline = TransformPipeline(transforms=[QuantileSorter(), transform])

    before_sorting = transform.transform(predictions)
    assert before_sorting.standard_deviation_series.isna().all()

    result = pipeline.transform(predictions)
    assert np.isnan(result.data["quantile_P10"].iloc[0])
    assert result.data["quantile_P50"].iloc[0] == 120.0
    assert result.data["forecast_mean"].iloc[0] == pytest.approx(123.33333333)
    assert result.standard_deviation_series.iloc[0] == pytest.approx(np.sqrt(188.88888889))
    assert np.isnan(result.data["forecast_mean"].iloc[1])
    assert np.isnan(result.standard_deviation_series.iloc[1])
    assert np.isfinite(result.standard_deviation_series.iloc[2])


def test_custom_mean_column_round_trips() -> None:
    predictions = ForecastDataset(
        pd.DataFrame(
            {"quantile_P10": [2.0], "quantile_P90": [6.0]},
            index=pd.date_range("2026-01-01", periods=1, tz="UTC", freq="15min"),
        ),
        forecast_mean_column="expected_load",
    )

    result = QuantileMomentsAdder().transform(predictions)
    round_trip = ForecastDataset(result.to_pandas())

    assert round_trip.forecast_mean_column == "expected_load"
    assert round_trip.data["expected_load"].iloc[0] == 4.0
    assert round_trip.quantiles == predictions.quantiles


@pytest.mark.parametrize(
    ("minimum", "expected_means", "expected_stdevs"),
    [
        pytest.param(2, [5.0, 4.0, np.nan, np.nan], [np.sqrt(19.0), 2.0, np.nan, np.nan], id="default"),
        pytest.param(3, [5.0, np.nan, np.nan, np.nan], [np.sqrt(19.0), np.nan, np.nan, np.nan], id="three-required"),
        pytest.param(4, [np.nan] * 4, [np.nan] * 4, id="insufficient-columns"),
    ],
)
def test_minimum_valid_quantiles(minimum: int, expected_means: list[float], expected_stdevs: list[float]) -> None:
    predictions = ForecastDataset(
        pd.DataFrame(
            {
                "quantile_P10": [0.0, 2.0, np.nan, 6.0],
                "quantile_P50": [4.0, np.nan, np.nan, np.nan],
                "quantile_P90": [12.0, 6.0, np.nan, 2.0],
            },
            index=pd.date_range("2026-01-01", periods=4, tz="UTC", freq="15min"),
        )
    )

    result = QuantileMomentsAdder(min_valid_quantiles=minimum).transform(predictions)

    np.testing.assert_allclose(result.data["forecast_mean"], expected_means)
    np.testing.assert_allclose(result.standard_deviation_series, expected_stdevs)


def test_minimum_valid_quantiles_rejects_one_point() -> None:
    with pytest.raises(ValueError, match="min_valid_quantiles must be at least 2"):
        QuantileMomentsAdder(min_valid_quantiles=1)
