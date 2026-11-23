# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_beam.evaluation.evaluation_helper import (
    compute_symmetric_quantile_metrics,
    rolling_quantile_reference_forecast,
)
from openstef_beam.evaluation.models import Filtering
from openstef_core.types import AvailableAt, LeadTime, Quantile


@pytest.mark.parametrize(
    ("reference_filtering", "forecast_time", "expected"),
    [
        # Cutoff 2025-05-02 06:00; context (04:00, 06:00] holds values 29 and 30.
        pytest.param(LeadTime.from_string("PT6H"), "2025-05-02T12:00:00", 29.5, id="lead_time"),
        # Cutoff 2025-05-01 06:00; context holds values 5 and 6.
        pytest.param(AvailableAt.from_string("D-1T0600"), "2025-05-02T12:00:00", 5.5, id="available_at"),
        # Cutoff 06:30 lies between grid points, so the context ending at 06:00 is used.
        pytest.param(LeadTime.from_string("PT6H"), "2025-05-02T12:30:00", 29.5, id="between_grid_cutoff"),
        # Cutoff 2025-05-02 03:00; context lacks the missing measurement at 03:00.
        pytest.param(LeadTime.from_string("PT9H"), "2025-05-02T12:00:00", np.nan, id="missing_measurement"),
        # Cutoff 2025-05-01 00:00; context holds only the first measurement.
        pytest.param(LeadTime.from_string("PT36H"), "2025-05-02T12:00:00", np.nan, id="partial_history"),
        # Cutoff 2025-04-30 12:00 lies before the first measurement.
        pytest.param(LeadTime.from_string("PT48H"), "2025-05-02T12:00:00", np.nan, id="no_history"),
    ],
)
def test_rolling_quantile_reference_forecast_returns_context_quantile_at_cutoff(
    reference_filtering: Filtering, forecast_time: str, expected: float
) -> None:
    # Arrange: hourly values equal to the hours since the start, with 2025-05-02 03:00 missing
    measurement_index = pd.date_range(datetime.fromisoformat("2025-05-01T00:00:00"), periods=49, freq="1h")
    measurements = pd.Series(np.arange(49, dtype=float), index=measurement_index).drop(
        datetime.fromisoformat("2025-05-02T03:00:00")
    )

    # Act
    reference = rolling_quantile_reference_forecast(
        measurements=measurements,
        index=pd.DatetimeIndex([datetime.fromisoformat(forecast_time)]),
        sample_interval=timedelta(hours=1),
        quantiles=[Quantile(0.5)],
        reference_filtering=reference_filtering,
        context_length=timedelta(hours=2),
        minimum_coverage=1.0,
    )

    # Assert
    np.testing.assert_array_equal(reference.quantiles_data.to_numpy(), [[expected]])


def test_compute_symmetric_quantile_metrics_uses_complementary_quantiles() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array(
        [
            [1.0, 5.0, 9.0],
            [2.0, 6.0, 10.0],
            [3.0, 7.0, 11.0],
        ]
    )
    quantiles = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]

    def metric(y_true: np.ndarray, y_pred_lower_q: np.ndarray, y_pred_upper_q: np.ndarray) -> float:
        assert y_true.tolist() == [10.0, 20.0, 30.0]
        return float(np.mean(y_pred_upper_q - y_pred_lower_q))

    result = compute_symmetric_quantile_metrics(
        y_true=y_true,
        y_pred=y_pred,
        quantiles=quantiles,
        metric_name="spread",
        metric=metric,
    )

    assert result == {
        Quantile(0.1): {"spread": 8.0},
        Quantile(0.9): {"spread": 8.0},
    }


def test_compute_symmetric_quantile_metrics_skips_missing_pairs_and_selected_quantiles() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array(
        [
            [1.0, 3.0, 7.0],
            [2.0, 4.0, 8.0],
            [3.0, 5.0, 9.0],
        ]
    )
    quantiles = [Quantile(0.1), Quantile(0.3), Quantile(0.7)]

    result = compute_symmetric_quantile_metrics(
        y_true=y_true,
        y_pred=y_pred,
        quantiles=quantiles,
        selected_quantiles=[Quantile(0.7)],
        metric_name="spread",
        metric=lambda y_true, y_pred_lower_q, y_pred_upper_q: float(np.mean(y_pred_upper_q - y_pred_lower_q)),
    )

    assert result == {Quantile(0.7): {"spread": 4.0}}


def test_compute_symmetric_quantile_metrics_passes_metric_kwargs() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array(
        [
            [1.0, 9.0],
            [2.0, 10.0],
            [3.0, 11.0],
        ]
    )
    quantiles = [Quantile(0.1), Quantile(0.9)]

    def metric(
        y_true: np.ndarray,
        y_pred_lower_q: np.ndarray,
        y_pred_upper_q: np.ndarray,
        multiplier: float,
    ) -> float:
        assert y_true.tolist() == [10.0, 20.0, 30.0]
        return float(np.mean(y_pred_upper_q - y_pred_lower_q) * multiplier)

    result = compute_symmetric_quantile_metrics(
        y_true=y_true,
        y_pred=y_pred,
        quantiles=quantiles,
        metric_name="weighted_spread",
        metric=metric,
        multiplier=0.5,
    )

    assert result == {
        Quantile(0.1): {"weighted_spread": 4.0},
        Quantile(0.9): {"weighted_spread": 4.0},
    }
