# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for QuantileCalibrationBoxPlotter - focuses on boxplot-specific functionality only.
Rest of functionality is tested in test_quantile_probability_plotter.py"""

import numpy as np

from openstef_beam.analysis.plots import QuantileCalibrationBoxPlotter
from openstef_core.types import Quantile


def test_plot_calculates_calibration_errors_correctly():
    """Test that plot correctly calculates calibration errors (observed - forecasted)"""
    # Arrange
    plotter = QuantileCalibrationBoxPlotter()
    # Model with varied forecasted and observed values
    plotter.add_model(
        "model1", [Quantile(0.1), Quantile(0.5), Quantile(0.9)], [Quantile(0.15), Quantile(0.7), Quantile(0.85)]
    )
    plotter.add_model(
        "model2", [Quantile(0.1), Quantile(0.5), Quantile(0.9)], [Quantile(0.12), Quantile(0.35), Quantile(0.92)]
    )

    # Act
    fig = plotter.plot()

    # Assert
    # Find the boxplot traces and verify calibration errors are calculated correctly
    box_traces = [trace for trace in fig.data if trace.type == "box"]

    # Find model1 trace
    model1_trace = next(trace for trace in box_traces if trace.name == "model1")
    model1_errors = list(model1_trace.y)
    expected_model1_errors = [0.05, 0.2, -0.05]
    assert np.allclose(model1_errors, expected_model1_errors), (
        f"Expected errors {expected_model1_errors}, got {model1_errors}"
    )

    # Find model2 trace
    model2_trace = next(trace for trace in box_traces if trace.name == "model2")
    model2_errors = list(model2_trace.y)
    expected_model2_errors = [0.02, -0.15, 0.02]
    assert np.allclose(model2_errors, expected_model2_errors), (
        f"Expected errors {expected_model2_errors}, got {model2_errors}"
    )


def test_plot_formats_quantile_labels_correctly():
    """Test that quantile levels are formatted as P## labels (e.g., P10, P50, P90)"""
    # Arrange
    plotter = QuantileCalibrationBoxPlotter()
    plotter.add_model(
        "model1",
        [Quantile(0.1), Quantile(0.25), Quantile(0.5), Quantile(0.75), Quantile(0.9)],
        [Quantile(0.11), Quantile(0.24), Quantile(0.51), Quantile(0.76), Quantile(0.94)],
    )

    # Act
    fig = plotter.plot()

    # Assert
    # Get x-axis tick labels to verify quantile formatting
    box_traces = [trace for trace in fig.data if trace.type == "box"]
    x_values = []
    for trace in box_traces:
        x_values.extend(trace.x)

    unique_x_values = set(x_values)
    expected_labels = {"P10", "P25", "P50", "P75", "P90"}
    assert unique_x_values == expected_labels


def test_plot_with_multiple_targets_creates_distributions():
    """Test that multiple targets per model create proper boxplot distributions"""
    # Arrange
    plotter = QuantileCalibrationBoxPlotter()
    quantiles = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]

    # Add multiple data points for same model (simulating different targets)
    plotter.add_model("ModelA", quantiles, [Quantile(0.08), Quantile(0.52), Quantile(0.88)])  # Target 1
    plotter.add_model("ModelA", quantiles, [Quantile(0.12), Quantile(0.48), Quantile(0.92)])  # Target 2
    plotter.add_model("ModelA", quantiles, [Quantile(0.09), Quantile(0.51), Quantile(0.89)])  # Target 3

    plotter.add_model("ModelB", quantiles, [Quantile(0.15), Quantile(0.55), Quantile(0.85)])  # Target 1
    plotter.add_model("ModelB", quantiles, [Quantile(0.18), Quantile(0.45), Quantile(0.82)])  # Target 2

    # Act
    fig = plotter.plot()

    # Assert
    box_traces = [trace for trace in fig.data if trace.type == "box"]

    # Should have 2 box traces (one per model)
    assert len(box_traces) == 2

    # Verify we have the right models
    model_a_traces = [trace for trace in box_traces if trace.name == "ModelA"]
    model_b_traces = [trace for trace in box_traces if trace.name == "ModelB"]

    assert len(model_a_traces) == 1  # One trace per model
    assert len(model_b_traces) == 1  # One trace per model

    # Each ModelA trace should have 9 data points (3 targets * 3 quantiles), ModelB should have 6
    model_a_trace = model_a_traces[0]
    model_b_trace = model_b_traces[0]
    assert len(model_a_trace.y) == 9, (
        f"ModelA should have 9 data points (3 targets * 3 quantiles), got {len(model_a_trace.y)}"
    )
    assert len(model_b_trace.y) == 6, (
        f"ModelB should have 6 data points (2 targets * 3 quantiles), got {len(model_b_trace.y)}"
    )
