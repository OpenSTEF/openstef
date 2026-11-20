# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import ForecastDataset
from openstef_core.mixins import TransformPipeline
from openstef_core.types import Quantile
from openstef_models.transforms.postprocessing import QuantileMoments, QuantileSorter


@pytest.mark.parametrize(
    ("levels", "values", "mean", "stdev"),
    [
        (
            [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            [70, 78, 88, 95, 110, 128, 140, 158, 175],
            113.41666667,
            25.93729083,
        ),
        (
            [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            [70, 78, 95, 110, 128, 158, 175],
            113.58333333,
            27.07743526,
        ),
        ([0.9, 0.1, 0.5], [120, 90, 100], 102.5, np.sqrt(118.75)),
        ([0.025, 0.997], [-4, 8], 2, 6),
        ([0.1, 0.7, 0.9], [0.1, 0.1, 0.1], 0.1, 0),
    ],
)
def test_moments_follow_the_available_grid(levels: list[float], values: list[float], mean: float, stdev: float) -> None:
    quantiles = [Quantile(level) for level in levels]
    index = pd.date_range("2026-01-01", periods=1, tz="UTC", freq="15min")
    predictions = pd.DataFrame([values], columns=[q.format() for q in quantiles], index=index)
    predictions["std"] = 999
    original = ForecastDataset(predictions, standard_deviation_column="std")

    result = QuantileMoments().transform(original)
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
    transform = QuantileMoments()
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
