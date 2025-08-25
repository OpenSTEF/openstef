# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go
import pytest

from openstef_beam.analysis.plots import PrecisionRecallCurvePlotter
from openstef_core.types import Q, Quantile


def test_add_model_valid_inputs():
    # Arrange
    plotter = PrecisionRecallCurvePlotter()
    model_name = "test_model"
    precision = [0.9, 0.8, 0.7]
    recall = [0.1, 0.2, 0.3]
    quantiles = [Quantile(0.1), Quantile(0.2), Quantile(0.3)]

    # Act
    result = plotter.add_model(model_name, precision, recall, quantiles)

    # Assert
    assert result is plotter  # Confirms method chaining
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model"] == model_name
    assert plotter.models_data[0]["precision"] == precision
    assert plotter.models_data[0]["recall"] == recall
    assert plotter.models_data[0]["quantile"] == quantiles


@pytest.mark.parametrize(
    ("precision", "recall", "quantiles", "error_msg"),
    [
        pytest.param(
            [0.9, 0.8],
            [0.1, 0.2, 0.3],
            [Quantile(0.1), Quantile(0.2), Quantile(0.3)],
            "Precision and recall lists must have the same length",
            id="precision_recall_mismatch",
        ),
        pytest.param(
            [0.9, 0.8, 0.7],
            [0.1, 0.2, 0.3],
            [Quantile(0.1), Quantile(0.2)],
            "Quantiles list must have the same length as precision and recall lists",
            id="quantiles_mismatch",
        ),
    ],
)
def test_add_model_invalid_inputs(
    precision: list[float], recall: list[float], quantiles: list[Quantile], error_msg: str
):
    # Arrange
    plotter = PrecisionRecallCurvePlotter()

    # Act & Assert
    with pytest.raises(ValueError, match=error_msg):
        plotter.add_model("test_model", precision, recall, quantiles)


def test_plot_no_models():
    # Arrange
    plotter = PrecisionRecallCurvePlotter()

    # Act & Assert
    with pytest.raises(ValueError, match=r"No model data has been added\. Use add_model first\."):
        plotter.plot()


def test_plot_with_models():
    # Arrange
    plotter = PrecisionRecallCurvePlotter()
    quantiles = [Quantile(0.1), Quantile(0.2)]
    plotter.add_model("model1", precision_values=[0.9, 0.8], recall_values=[0.1, 0.2], quantiles=quantiles)
    plotter.add_model("model2", precision_values=[0.7, 0.6], recall_values=[0.3, 0.4], quantiles=quantiles)

    # Act
    fig = plotter.plot(title="Test Plot")

    # Assert
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # type: ignore
    assert fig.layout.title.text == "Test Plot"  # type: ignore


def test_model_chaining():
    # Arrange
    plotter = PrecisionRecallCurvePlotter()

    # Act
    result = plotter.add_model("model1", precision_values=[0.9], recall_values=[0.1], quantiles=[Q(0.1)]).add_model(
        "model2", precision_values=[0.8], recall_values=[0.2], quantiles=[Q(0.2)]
    )

    # Assert
    assert result is plotter
    assert len(plotter.models_data) == 2
    assert plotter.models_data[0]["model"] == "model1"
    assert plotter.models_data[1]["model"] == "model2"
