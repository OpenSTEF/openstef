# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Sequence

import numpy as np
import pytest

from openstef_beam.metrics import (
    ConfusionMatrix,
    PrecisionRecall,
    confusion_matrix,
    fbeta,
    mape,
    precision_recall,
    rmae,
)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "lower_quantile", "upper_quantile", "sample_weights", "expected", "tol"),
    [
        pytest.param([1, 2, 3], [1, 2, 3], 0.05, 0.95, None, 0.0, 1e-8, id="identical_arrays"),
        pytest.param([1, 2, 3], [1, 2, 5.7], 0.05, 0.95, None, 0.5, 1e-8, id="rmae_exact_half"),
        pytest.param([1, 1, 1], [1, 1, 1], 0.05, 0.95, None, np.nan, 0, id="constant_array_nan"),
        pytest.param([1, 2, 3], [2, 3, 4], 0.0, 1.0, None, 0.5, 1e-8, id="custom_quantiles"),
        pytest.param(
            [1, 2, 3],
            [1, 2, 3],
            0.05,
            0.95,
            [1.0, 1.0, 1.0],
            0.0,
            1e-8,
            id="identical_arrays_sample_weighted",
        ),
    ],
)
def test_rmae_various(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    lower_quantile: float,
    upper_quantile: float,
    sample_weights: Sequence[float] | None,
    expected: float,
    tol: float,
) -> None:
    # Arrange
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Prepare weights argument for rmae
    weights_arg = np.array(sample_weights) if sample_weights is not None else None

    # Act
    result = rmae(
        y_true_arr,
        y_pred_arr,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        sample_weights=weights_arg,
    )

    # Assert
    if np.isnan(expected):
        assert np.isnan(result), f"Expected NaN but got {result}"
    else:
        assert abs(result - expected) < tol, f"Expected {expected} but got {result}"


@pytest.mark.parametrize(
    ("y_true", "y_pred", "sample_weights", "expected", "tol"),
    [
        pytest.param(
            [0, 0, 10],
            [0, 10, 0],
            None,
            2 / 3,
            1e-8,
            id="unweighted_two_thirds",
        ),
        pytest.param(
            [0, 0, 10],
            [0, 10, 0],
            [1.0, 1.0, 1.0],
            2 / 3,
            1e-8,
            id="all_ones_equals_unweighted",
        ),
        pytest.param(
            [0, 0, 10],
            [0, 10, 0],
            [1.0, 0.0, 0.0],
            0.0,
            1e-8,
            id="only_first_sample_weighted_zero_rmae",
        ),
        pytest.param(
            [0, 0, 10],
            [0, 10, 0],
            [0.0, 0.0, 1.0],
            1.0,
            1e-8,
            id="only_last_sample_weighted_full_rmae",
        ),
    ],
)
def test_rmae_sample_weights_behavior(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    sample_weights: Sequence[float] | None,
    expected: float,
    tol: float,
) -> None:
    # Arrange
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    weights_arg = np.array(sample_weights) if sample_weights is not None else None

    # Act
    # use full range quantiles so range = max - min = 10 in these vectors
    result = rmae(y_true_arr, y_pred_arr, lower_quantile=0.0, upper_quantile=1.0, sample_weights=weights_arg)

    # Assert
    # Check numerical equality to the expected concrete value
    assert abs(result - expected) < tol, f"Expected {expected} but got {result} for weights={sample_weights}"


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected", "tol"),
    [
        pytest.param([1, 2, 3], [1, 2, 3], 0.0, 1e-8, id="identical_arrays"),
        pytest.param([2, 4, 10], [3, 6, 15], 0.5, 1e-8, id="exact_50_percent"),
        pytest.param([10, 20, 30], [5, 10, 15], 0.5, 1e-8, id="exact_minus_50_percent"),
    ],
)
def test_mape_various(y_true: Sequence[float], y_pred: Sequence[float], expected: float, tol: float) -> None:
    # Arrange
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Act
    result = mape(y_true_arr, y_pred_arr)

    # Assert
    assert abs(result - expected) < tol, f"Expected {expected} but got {result}"


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param([0, 2, 4], [1, 2, 4], id="first_value_zero"),
        pytest.param([0, 0, 0], [1, 1, 1], id="all_zeros"),
    ],
)
def test_mape_division_by_zero(y_true: Sequence[float], y_pred: Sequence[float]) -> None:
    # Arrange
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Act & Assert
    with pytest.warns(RuntimeWarning):
        result = mape(y_true_arr, y_pred_arr)

    assert np.isnan(result) or np.isinf(result)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_result"),
    [
        pytest.param(
            [-1, 5, 2],
            [-1, 5, 2],
            {
                "tp": [True, True, False],
                "tn": [False, False, True],
                "fp": [False, False, False],
                "fn": [False, False, False],
            },
            id="perfect_match",
        ),
        pytest.param(
            [0, 5, 2],
            [5, 0, 2],
            {
                "tp": [False, False, False],
                "tn": [False, False, True],
                "fp": [True, False, False],
                "fn": [False, True, False],
            },
            id="swapped_extremes",
        ),
    ],
)
def test_confusion_matrix_simple(
    y_true: Sequence[float], y_pred: Sequence[float], expected_result: dict[str, Sequence[bool]]
) -> None:
    # Arrange
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Act
    cm = confusion_matrix(y_true_arr, y_pred_arr, limit_pos=3, limit_neg=-1)

    # Assert
    assert np.array_equal(cm.true_positives, np.array(expected_result["tp"]))
    assert np.array_equal(cm.true_negatives, np.array(expected_result["tn"]))
    assert np.array_equal(cm.false_positives, np.array(expected_result["fp"]))
    assert np.array_equal(cm.false_negatives, np.array(expected_result["fn"]))


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_effective"),
    [
        pytest.param([2, 4, 0], [2, 5, -1], [False, True, True], id="effective_prediction"),
        pytest.param([2, 4, 0], [2, 3, -1], [False, False, True], id="ineffective_prediction"),
    ],
)
def test_confusion_matrix_effective(
    y_true: Sequence[float], y_pred: Sequence[float], expected_effective: Sequence[bool]
) -> None:
    # Arrange
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Act
    cm = confusion_matrix(y_true_arr, y_pred_arr, limit_pos=3, limit_neg=1)

    # Assert
    assert np.array_equal(cm.effective_true_positives, np.array(expected_effective))


@pytest.mark.parametrize(
    ("cm", "effective", "expected"),
    [
        pytest.param(
            ConfusionMatrix(
                true_positives=np.array([True, True]),
                true_negatives=np.array([True, True]),
                false_positives=np.array([False, False]),
                false_negatives=np.array([False, False]),
                effective_true_positives=np.array([True, True]),
                ineffective_true_positives=np.array([False, False]),
            ),
            False,
            {"precision": 1.0, "recall": 1.0},
            id="perfect_prediction",
        ),
        pytest.param(
            ConfusionMatrix(
                true_positives=np.array([True, False]),
                true_negatives=np.array([False, True]),
                false_positives=np.array([True, False]),
                false_negatives=np.array([False, False]),
                effective_true_positives=np.array([True, False]),
                ineffective_true_positives=np.array([False, False]),
            ),
            False,
            {"precision": 0.5, "recall": 1.0},
            id="half_precision",
        ),
    ],
)
def test_precision_recall_simple(cm: ConfusionMatrix, effective: bool, expected: dict[str, float]) -> None:
    # Act
    pr = precision_recall(cm, effective=effective)

    # Assert
    assert pr.precision == expected["precision"], (
        f"Precision mismatch: got {pr.precision}, expected {expected['precision']}"
    )
    assert pr.recall == expected["recall"], f"Recall mismatch: got {pr.recall}, expected {expected['recall']}"


@pytest.mark.parametrize(
    ("precision", "recall", "beta", "expected"),
    [
        pytest.param(1.0, 1.0, 2.0, 1.0, id="perfect_scores"),
        pytest.param(0.0, 0.0, 2.0, 0.0, id="zero_scores"),
        pytest.param(0.5, 1.0, 1.0, 2 / 3, id="f1"),
        pytest.param(0.5, 1.0, 2.0, 5 / 6, id="f2"),
    ],
)
def test_fbeta_simple(precision: float, recall: float, beta: float, expected: float) -> None:
    # Arrange
    pr = PrecisionRecall(precision=precision, recall=recall)

    # Act
    score = fbeta(pr, beta=beta)

    # Assert
    assert abs(score - expected) < 1e-8, f"Expected {expected} but got {score}"
