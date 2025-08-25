# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go
import pytest

from openstef_beam.analysis.plots import GroupedTargetMetricPlotter


def test_add_model():
    # Arrange
    plotter = GroupedTargetMetricPlotter()
    model_name = "TestModel"
    targets = ["Target1", "Target2"]
    metrics = [0.9, 0.8]

    # Act
    result = plotter.add_model(model_name, targets, metrics)

    # Assert
    assert result is plotter  # Return value should be self for method chaining
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model"] == model_name
    assert plotter.models_data[0]["target"] == targets
    assert plotter.models_data[0]["metric_value"] == metrics


def test_add_model_with_mismatched_lengths():
    # Arrange
    plotter = GroupedTargetMetricPlotter()
    model_name = "TestModel"
    targets = ["Target1", "Target2"]
    metrics = [0.9]  # Only one metric for two targets

    # Act & Assert
    with pytest.raises(ValueError, match="Targets and metric values must have the same length"):
        plotter.add_model(model_name, targets, metrics)


def test_set_target_groups():
    # Arrange
    plotter = GroupedTargetMetricPlotter()
    target_groups = {"Target1": "Group1", "Target2": "Group2"}

    # Act
    result = plotter.set_target_groups(target_groups)

    # Assert
    assert result is plotter  # Return value should be self for method chaining
    assert plotter.target_groups == target_groups


@pytest.mark.parametrize(
    ("use_target_groups", "expected_plot_type"),
    [
        pytest.param(False, "bar", id="without_groups"),
        pytest.param(True, "box", id="with_groups"),
    ],
)
def test_plot_creates_correct_figure_type(use_target_groups: bool, expected_plot_type: str):
    # Arrange
    plotter = GroupedTargetMetricPlotter()
    plotter.add_model("Model1", ["Target1", "Target2"], [0.9, 0.8])

    if use_target_groups:
        plotter.set_target_groups({"Target1": "Group1", "Target2": "Group1"})

    # Act
    fig = plotter.plot(title="Test Plot", metric_name="Accuracy")

    # Assert
    assert isinstance(fig, go.Figure)
    if expected_plot_type == "bar":
        assert any(trace.type == "bar" for trace in fig.data)
    elif expected_plot_type == "box":
        assert any(trace.type == "box" for trace in fig.data)


def test_plot_with_multiple_models():
    # Arrange
    plotter = GroupedTargetMetricPlotter()
    plotter.add_model("Model1", ["Target1", "Target2"], [0.9, 0.8])
    plotter.add_model("Model2", ["Target1", "Target2"], [0.7, 0.6])

    # Act
    fig = plotter.plot(title="Test Plot", metric_name="Accuracy")

    # Assert
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two traces for two models


def test_plot_with_no_data():
    # Arrange
    plotter = GroupedTargetMetricPlotter()

    # Act & Assert
    with pytest.raises(ValueError, match=r"No model data has been added. Use add_model first."):
        plotter.plot()


def test_plot_with_missing_target_in_group_mapping():
    # Arrange
    plotter = GroupedTargetMetricPlotter()
    plotter.add_model("Model1", ["Target1", "Target2", "Target3"], [0.9, 0.8, 0.7])
    # Missing Target3 in the mapping
    plotter.set_target_groups({"Target1": "Group1", "Target2": "Group1"})

    # Act & Assert
    with pytest.raises(ValueError, match="Some targets are missing from the target group mapping"):
        plotter.plot()
