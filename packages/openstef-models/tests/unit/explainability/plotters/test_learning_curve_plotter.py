# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for LearningCurvePlotter."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from openstef_core.exceptions import MissingColumnsError
from openstef_models.explainability.plotters.learning_curve_plotter import LearningCurvePlotter


@pytest.fixture
def curves_with_validation() -> pd.DataFrame:
    """Validation error is lowest at round 3, then rises, which is a typical overfitting pattern."""
    return pd.DataFrame(
        {"train": [5.0, 4.0, 3.0, 2.0, 1.0], "validation": [5.5, 4.5, 4.0, 4.2, 4.6]},
        index=pd.RangeIndex(start=1, stop=6, name="round"),
    )


def test_plot__returns_figure(curves_with_validation: pd.DataFrame) -> None:
    # Act
    figure = LearningCurvePlotter.plot(curves_with_validation)

    # Assert
    assert isinstance(figure, go.Figure)


def test_plot__draws_train_validation_and_best_marker(curves_with_validation: pd.DataFrame) -> None:
    # Act
    figure = LearningCurvePlotter.plot(curves_with_validation)

    # Assert
    assert [trace.name for trace in figure.data] == ["train", "validation", "best validation (round 3)"]
    assert list(figure.data[0].x) == [1, 2, 3, 4, 5]
    assert list(figure.data[0].y) == [5.0, 4.0, 3.0, 2.0, 1.0]
    assert list(figure.data[1].y) == [5.5, 4.5, 4.0, 4.2, 4.6]


def test_plot__best_marker_is_at_lowest_validation_error(curves_with_validation: pd.DataFrame) -> None:
    # Act
    figure = LearningCurvePlotter.plot(curves_with_validation)

    # Assert
    best_marker = figure.data[-1]
    assert list(best_marker.x) == [3]
    assert list(best_marker.y) == [4.0]


def test_plot__without_validation_draws_only_train_line() -> None:
    # Arrange
    curves = pd.DataFrame({"train": [3.0, 2.0, 1.0]}, index=pd.RangeIndex(start=1, stop=4, name="round"))

    # Act
    figure = LearningCurvePlotter.plot(curves)

    # Assert
    assert [trace.name for trace in figure.data] == ["train"]


def test_plot__missing_train_column_raises_error() -> None:
    # Arrange
    curves = pd.DataFrame({"validation": [3.0, 2.0, 1.0]}, index=pd.RangeIndex(start=1, stop=4, name="round"))

    # Act & Assert
    with pytest.raises(MissingColumnsError):
        LearningCurvePlotter.plot(curves)
