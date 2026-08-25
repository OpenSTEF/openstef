# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from openstef_beam.evaluation.evaluation_helper import compute_symmetric_quantile_metrics
from openstef_core.types import Quantile


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
