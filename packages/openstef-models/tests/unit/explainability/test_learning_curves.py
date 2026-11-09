# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for build_learning_curves."""

from collections import OrderedDict

import numpy as np
import pandas as pd

from openstef_models.explainability.mixins import build_learning_curves


def test_build_learning_curves__train_and_validation_mapped_by_position() -> None:
    # Arrange
    # XGBoost names its sets validation_0 and validation_1, LightGBM names them valid_0 and valid_1.
    xgboost_history = {
        "validation_0": OrderedDict({"mean_pinball_loss": [3.0, 2.0, 1.0]}),
        "validation_1": OrderedDict({"mean_pinball_loss": [4.0, 3.5, 3.0]}),
    }
    lightgbm_history = {
        "valid_0": OrderedDict({"quantile": [3.0, 2.0, 1.0]}),
        "valid_1": OrderedDict({"quantile": [4.0, 3.5, 3.0]}),
    }

    # Act
    xgboost_curves = build_learning_curves(xgboost_history)
    lightgbm_curves = build_learning_curves(lightgbm_history)

    # Assert
    expected = pd.DataFrame(
        {"train": [3.0, 2.0, 1.0], "validation": [4.0, 3.5, 3.0]},
        index=pd.RangeIndex(start=1, stop=4, name="round"),
    )
    pd.testing.assert_frame_equal(xgboost_curves, expected)
    pd.testing.assert_frame_equal(lightgbm_curves, expected)


def test_build_learning_curves__without_validation_has_only_train_column() -> None:
    # Arrange
    history = {"validation_0": {"mean_pinball_loss": [3.0, 2.0]}}

    # Act
    curves = build_learning_curves(history)

    # Assert
    assert list(curves.columns) == ["train"]
    assert list(curves["train"]) == [3.0, 2.0]


def test_build_learning_curves__index_starts_at_one_and_is_named_round() -> None:
    # Arrange
    history = {"validation_0": {"mean_pinball_loss": [0.5] * 7}}

    # Act
    curves = build_learning_curves(history)

    # Assert
    assert list(curves.index) == [1, 2, 3, 4, 5, 6, 7]
    assert curves.index.name == "round"


def test_build_learning_curves__converts_numpy_values_to_float() -> None:
    # Arrange
    # LightGBM returns numpy floats in its history.
    history = {"valid_0": {"quantile": [np.float64(0.5), np.float64(0.25)]}}

    # Act
    curves = build_learning_curves(history)

    # Assert
    assert curves["train"].dtype == np.float64
    assert list(curves["train"]) == [0.5, 0.25]


def test_build_learning_curves__uses_first_metric_when_several_are_recorded() -> None:
    # Arrange
    history = {"validation_0": {"first_metric": [1.0, 0.5], "second_metric": [9.0, 9.0]}}

    # Act
    curves = build_learning_curves(history)

    # Assert
    assert list(curves["train"]) == [1.0, 0.5]


def test_build_learning_curves__ignores_sets_beyond_train_and_validation() -> None:
    # Arrange
    history = {
        "validation_0": {"m": [1.0]},
        "validation_1": {"m": [2.0]},
        "validation_2": {"m": [3.0]},
    }

    # Act
    curves = build_learning_curves(history)

    # Assert
    assert list(curves.columns) == ["train", "validation"]
