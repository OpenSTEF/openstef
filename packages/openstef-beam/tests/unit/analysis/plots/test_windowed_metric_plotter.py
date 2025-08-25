# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime

import plotly.graph_objects as go
import pytest

from openstef_beam.analysis.plots import WindowedMetricPlotter


@pytest.fixture
def sample_datetime_series():
    """Returns a small series of datetime objects."""
    return [
        datetime.fromisoformat("2023-01-01T00:00:00"),
        datetime.fromisoformat("2023-01-02T00:00:00"),
        datetime.fromisoformat("2023-01-03T00:00:00"),
    ]


def test_add_model(sample_datetime_series: list[datetime]):
    # Arrange
    plotter = WindowedMetricPlotter()
    model_name = "test_model"
    metric_values = [0.1, 0.2, 0.3]

    # Act
    result = plotter.add_model(model_name, sample_datetime_series, metric_values)

    # Assert
    assert result is plotter  # Tests method chaining
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model"] == model_name
    assert plotter.models_data[0]["timestamp"] == sample_datetime_series
    assert plotter.models_data[0]["metric_value"] == metric_values


def test_add_model_validates_input_lengths(sample_datetime_series: list[datetime]):
    # Arrange
    plotter = WindowedMetricPlotter()
    model_name = "test_model"
    metric_values = [0.1, 0.2]  # Only 2 values vs 3 timestamps

    # Act & Assert
    with pytest.raises(ValueError, match="Timestamps and metric values must have the same length"):
        plotter.add_model(model_name, sample_datetime_series, metric_values)


def test_set_window_size():
    # Arrange
    plotter = WindowedMetricPlotter()
    window_size = "7 days"

    # Act
    result = plotter.set_window_size(window_size)

    # Assert
    assert result is plotter  # Tests method chaining
    assert plotter.window_size == window_size


def test_plot_with_no_data():
    # Arrange
    plotter = WindowedMetricPlotter()

    # Act & Assert
    with pytest.raises(ValueError, match=r"No model data has been added. Use add_model first."):
        plotter.plot()


def test_plot(sample_datetime_series: list[datetime]):
    # Arrange
    plotter = WindowedMetricPlotter()
    model_name = "test_model"
    metric_name = "Test Metric"
    plot_title = "Test Plot"
    window_size = "7 days"
    metric_values = [0.1, 0.2, 0.3]
    plotter.add_model(model_name, sample_datetime_series, metric_values)
    plotter.set_window_size(window_size)

    # Act
    fig = plotter.plot(title=plot_title, metric_name=metric_name)

    # Assert
    assert isinstance(fig, go.Figure)
    assert plot_title in fig.layout.title.text  # type: ignore - needs stubs
    assert window_size in fig.layout.title.text  # type: ignore - needs stubs
    assert metric_name in fig.layout.yaxis.title.text  # type: ignore - needs stubs
