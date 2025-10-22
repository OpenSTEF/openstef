# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
# pyright: basic, reportAttributeAccessIssue=false

from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_beam.analysis.plots.forecast_time_series_plotter import BandData, QuantilePolygonStyle


def test_add_model_with_forecast_only():
    # Arrange
    forecast = pd.Series(data=[1, 2, 3], index=pd.date_range("2023-01-01", periods=3, freq="D"))

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model A", forecast=forecast)

    # Assert
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model_name"] == "Model A"
    assert plotter.models_data[0]["forecast"] is not None
    pd.testing.assert_series_equal(plotter.models_data[0]["forecast"], forecast)
    assert plotter.models_data[0]["quantiles"] is None


def test_add_model_with_quantiles_only():
    # Arrange
    quantiles = pd.DataFrame(
        data={"quantile_P10": [1, 2], "quantile_P90": [3, 4]},
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
    )

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model B", quantiles=quantiles)

    # Assert
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model_name"] == "Model B"
    assert plotter.models_data[0]["forecast"] is None
    assert plotter.models_data[0]["quantiles"] is not None
    pd.testing.assert_frame_equal(plotter.models_data[0]["quantiles"], quantiles)


def test_add_model_with_forecast_and_quantiles():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="D")

    forecast = pd.Series(
        data=[1, 2, 3],
        index=index,
    )
    quantiles = pd.DataFrame(data={"quantile_P10": [1, 2, 3], "quantile_P90": [4, 5, 6]}, index=index)

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model C", forecast=forecast, quantiles=quantiles)

    # Assert
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model_name"] == "Model C"
    assert plotter.models_data[0]["forecast"] is not None
    pd.testing.assert_series_equal(plotter.models_data[0]["forecast"], forecast)
    assert plotter.models_data[0]["quantiles"] is not None
    pd.testing.assert_frame_equal(plotter.models_data[0]["quantiles"], quantiles)


def test_method_chaining():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    measurements = pd.Series([10, 20, 30], index=index)
    forecast1 = pd.Series([1, 2, 3], index=index)
    forecast2 = pd.Series([4, 5, 6], index=index)

    # Act - test method chaining
    plotter = (
        ForecastTimeSeriesPlotter()
        .add_measurements(measurements)
        .add_model(model_name="Model 1", forecast=forecast1)
        .add_model(model_name="Model 2", forecast=forecast2)
    )

    # Assert
    assert plotter.measurements is not None
    pd.testing.assert_series_equal(plotter.measurements, measurements)
    assert len(plotter.models_data) == 2
    assert plotter.models_data[0]["model_name"] == "Model 1"
    assert plotter.models_data[1]["model_name"] == "Model 2"


def test_add_model_raises_if_no_data():
    # Arrange
    plotter = ForecastTimeSeriesPlotter()

    # Act & Assert
    with pytest.raises(ValueError, match=r"At least one of forecast or quantiles must be provided."):
        plotter.add_model(model_name="Model X")


def test_add_model_raises_if_quantile_column_wrong_format():
    # Arrange
    quantiles = pd.DataFrame(
        {"bad_column": [1, 2], "quantile_P90": [3, 4]},
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
    )
    plotter = ForecastTimeSeriesPlotter()

    # Act & Assert
    with pytest.raises(ValueError, match="Column 'bad_column' does not follow the expected format"):
        plotter.add_model(model_name="Model Y", quantiles=quantiles)


def test_add_measurements():
    # Arrange
    measurements = pd.Series([10, 20, 30], index=pd.date_range("2024-01-01", periods=3, freq="D"), name="measurement")

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_measurements(measurements)

    # Assert
    assert plotter.measurements is not None
    pd.testing.assert_series_equal(plotter.measurements, measurements)


def test_plot_with_no_data_raises():
    # Arrange
    plotter = ForecastTimeSeriesPlotter()

    # Act & Assert
    with pytest.raises(ValueError, match=r"No data has been added. Use add_measurements or add_model first."):
        plotter.plot()


def test_plot_with_only_measurements():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    measurements = pd.Series([10, 20, 30], index=index, name="value")

    plotter = ForecastTimeSeriesPlotter()
    plotter.add_measurements(measurements)

    # Act
    fig = plotter.plot()

    # Assert
    assert isinstance(fig, go.Figure)
    # Should have one trace for the measurements
    assert len(fig.data) == 1  # pyright: ignore[reportArgumentType, reportUnknownMemberType]


def test_plot_with_forecasts_and_no_quantiles():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    forecast1 = pd.Series([1, 2, 3], index=index, name="value")
    forecast2 = pd.Series([4, 5, 6], index=index, name="value")

    plotter = (
        ForecastTimeSeriesPlotter()
        .add_model(model_name="Model 1", forecast=forecast1)
        .add_model(model_name="Model 2", forecast=forecast2)
    )

    # Act
    fig = plotter.plot()

    # Assert
    # Should have two traces, one for each forecast
    assert len(fig.data) == 2  # pyright: ignore[reportArgumentType, reportUnknownMemberType]


def test_plot_with_measurements_and_forecasts():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    measurements = pd.Series([10, 20, 30], index=index, name="value")
    forecast = pd.Series([1, 2, 3], index=index, name="value")

    plotter = (
        ForecastTimeSeriesPlotter().add_measurements(measurements).add_model(model_name="Model 1", forecast=forecast)
    )

    # Act
    fig = plotter.plot()

    # Assert
    # Should have two traces: measurements and forecast
    assert len(fig.data) == 2  # pyright: ignore[reportArgumentType, reportUnknownMemberType]


def test_plot_with_forecasts_and_quantiles():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    forecast = pd.Series([1, 2, 3], index=index, name="value")
    quantiles = pd.DataFrame(
        {"quantile_P10": [0, 1, 2], "quantile_P50": [1, 2, 3], "quantile_P90": [2, 3, 4]},
        index=index,
    )

    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model 1", forecast=forecast, quantiles=quantiles)

    # Act
    fig = plotter.plot()

    # Assert
    # Should have forecast trace and quantile bands (each band adds 2 traces)
    # For 10/90 quantile, one band (2 traces), plus forecast (1 trace): total 3
    assert len(fig.data) == 3  # type: ignore - needs stubs
    names: list[str] = [trace.name for trace in fig.data]  # type: ignore - needs stubs
    assert any("Forecast (50th)" in n for n in names)
    assert any("10%-90%" in n for n in names)


def test_plot_with_measurements_forecasts_and_quantiles_multiple_models():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    measurements = pd.Series([10, 20, 30], index=index, name="value")

    forecast1 = pd.Series([1, 2, 3], index=index, name="value")

    quantiles1 = pd.DataFrame(
        {"quantile_P10": [0, 1, 2], "quantile_P50": [1, 2, 3], "quantile_P90": [2, 3, 4]},
        index=index,
    )

    forecast2 = pd.Series([4, 5, 6], index=index, name="value")
    quantiles2 = pd.DataFrame(
        {"quantile_P10": [3, 4, 5], "quantile_P50": [4, 5, 6], "quantile_P90": [5, 6, 7]},
        index=index,
    )

    plotter = (
        ForecastTimeSeriesPlotter()
        .add_measurements(measurements)
        .add_model(model_name="Model 1", forecast=forecast1, quantiles=quantiles1)
        .add_model(model_name="Model 2", forecast=forecast2, quantiles=quantiles2)
    )

    # Act
    fig = plotter.plot()

    # Assert
    # 1 measurement trace, 2 forecast traces, 2*2 quantile traces (one band per model, each band = 2 traces)
    assert len(fig.data) == 7  # pyright: ignore[reportArgumentType, reportUnknownMemberType]


def test_custom_title():
    # Arrange
    measurements = pd.Series(
        data=[10, 20, 30],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
        name="measurement",
    )
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_measurements(measurements)

    # Act
    fig = plotter.plot(title="Custom Title")

    # Assert
    assert cast(str, fig.layout.title.text) == "Custom Title"  # type: ignore - needs stubs


def test_plot_with_only_quantiles_includes_50th():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    quantiles = pd.DataFrame(
        {"quantile_P10": [0, 1, 2], "quantile_P50": [1, 2, 3], "quantile_P90": [2, 3, 4]},
        index=index,
    )

    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model 1", quantiles=quantiles)

    # Act
    fig = plotter.plot()

    # Assert
    # Should have quantile bands (2 traces for 10/90) plus 50th quantile line (1 trace): total 3
    assert len(fig.data) == 3  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
    names: list[str] = [trace.name for trace in fig.data]  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue, reportUnknownMemberType]
    assert any("Quantile (50th)" in n for n in names)
    assert any("10%-90%" in n for n in names)


@pytest.mark.parametrize("connect_gaps", [True, False])
def test_insert_gaps_for_missing_timestamps(connect_gaps: bool):
    """Test the _insert_gaps_for_missing_timestamps method."""
    # Arrange
    plotter = ForecastTimeSeriesPlotter(connect_gaps=connect_gaps)

    # Create example time series with gaps
    dates_with_gaps = pd.to_datetime([
        "2024-01-01 10:00:00",
        "2024-01-01 11:00:00",
        "2024-01-01 13:00:00",
        "2024-01-01 14:00:00",
        "2024-01-01 18:00:00",
        "2024-01-01 19:00:00",
    ])
    original_series = pd.Series([10, 20, 30, 40, 50, 60], index=dates_with_gaps)
    sample_interval = pd.Timedelta("1h")

    # Act
    result = plotter._insert_gaps_for_missing_timestamps(original_series, sample_interval)

    # Assert
    if connect_gaps:
        # connect_gaps=True should return unchanged series
        pd.testing.assert_series_equal(result, original_series)
    else:
        # connect_gaps=False should have filled in missing timestamps with NaN
        expected_index = pd.date_range(start="2024-01-01 10:00:00", end="2024-01-01 19:00:00", freq="1h")
        expected_values = [10, 20, np.nan, 30, 40, np.nan, np.nan, np.nan, 50, 60]
        expected_series = pd.Series(expected_values, index=expected_index)

        pd.testing.assert_series_equal(result, expected_series)


def test_add_segmented_quantile_polygons():
    """Test the _add_segmented_quantile_polygons method directly."""
    # Arrange
    plotter = ForecastTimeSeriesPlotter(connect_gaps=False)
    figure = go.Figure()

    # Create example indices
    complete_index = pd.date_range(start="2024-01-01 10:00:00", end="2024-01-01 18:00:00", freq="1h")

    # Create data with explicit NaN values for gaps
    lower_values = [5, 10, 15, np.nan, np.nan, 20, 25, np.nan, 30]
    upper_values = [15, 25, 35, np.nan, np.nan, 45, 55, np.nan, 65]

    lower_quantile_data = pd.Series(lower_values, index=complete_index)
    upper_quantile_data = pd.Series(upper_values, index=complete_index)

    band = BandData(
        model_name="TestModel",
        model_index=0,
        lower_quantile=10,
        upper_quantile=90,
        lower_data=lower_quantile_data,
        upper_data=upper_quantile_data,
    )

    style = QuantilePolygonStyle(
        fill_color="rgb(0, 0, 255)",
        stroke_color="rgb(0, 0, 200)",
        legendgroup="test_group",
    )

    # Act
    plotter._add_segmented_quantile_polygons(figure, band, style)

    # Assert
    polygon_traces = [trace for trace in figure.data if hasattr(trace, "fill") and trace.fill == "toself"]

    # Should create 3 separate polygon traces for the 3 continuous segments
    assert len(polygon_traces) == 3, f"Expected 3 polygon segments, got {len(polygon_traces)}"

    # Verify each segment has correct structure with different lengths: 3, 2, 1 points
    expected_lengths = [6, 4, 2]  # 3-point, 2-point, 1-point segments -> 6, 4, 2 polygon coordinates
    for i, polygon_trace in enumerate(polygon_traces):
        # Each segment polygon should have: n_points * 2 coordinates (lower + upper reversed)
        assert len(polygon_trace.x) == expected_lengths[i], (
            f"Segment {i} should have {expected_lengths[i]} x-coordinates, got {len(polygon_trace.x)}"
        )
        assert len(polygon_trace.y) == expected_lengths[i], (
            f"Segment {i} should have {expected_lengths[i]} y-coordinates, got {len(polygon_trace.y)}"
        )

    # Verify segment coordinates match expected continuous data ranges
    expected_segments = [
        {
            "x_times": ["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 12:00:00"],
            "y_lower": [5, 10, 15],
            "y_upper": [15, 25, 35],
        },
        {"x_times": ["2024-01-01 15:00:00", "2024-01-01 16:00:00"], "y_lower": [20, 25], "y_upper": [45, 55]},
        {"x_times": ["2024-01-01 18:00:00"], "y_lower": [30], "y_upper": [65]},
    ]

    for i, (polygon_trace, expected) in enumerate(zip(polygon_traces, expected_segments, strict=True)):
        # Convert expected times to pandas timestamps for comparison
        expected_x = [pd.Timestamp(t) for t in expected["x_times"]]

        # Polygon should have: [x1, x2, ...] + [x_n, ..., x1] (lower forward + upper backward)
        expected_polygon_x = expected_x + expected_x[::-1]
        expected_polygon_y = expected["y_lower"] + expected["y_upper"][::-1]

        assert list(polygon_trace.x) == expected_polygon_x, f"Segment {i} x-coordinates mismatch"
        assert list(polygon_trace.y) == expected_polygon_y, f"Segment {i} y-coordinates mismatch"


@pytest.mark.parametrize("connect_gaps", [True, False])
def test_add_quantile_band(connect_gaps: bool):
    """Test the _add_quantile_band method with both connect_gaps modes."""
    # Arrange
    plotter = ForecastTimeSeriesPlotter(connect_gaps=connect_gaps)
    figure = go.Figure()

    # Create example quantile data with gaps
    dates_with_gaps = pd.to_datetime([
        "2024-01-01 10:00:00",
        "2024-01-01 11:00:00",
        "2024-01-01 13:00:00",
        "2024-01-01 14:00:00",
        "2024-01-01 18:00:00",
        "2024-01-01 19:00:00",
    ])

    lower_quantile_data = pd.Series([5, 10, 15, 20, 25, 30], index=dates_with_gaps)
    upper_quantile_data = pd.Series([15, 25, 35, 45, 55, 65], index=dates_with_gaps)

    # Act
    plotter._add_quantile_band(
        figure=figure,
        lower_quantile_data=lower_quantile_data,
        lower_quantile=10.0,
        upper_quantile_data=upper_quantile_data,
        upper_quantile=90.0,
        model_name="TestModel",
        model_index=0,
    )

    # Assert
    # Should have added traces to the figure
    assert len(figure.data) > 0  # type: ignore[attr-defined]

    # Get polygon traces (filled areas)
    polygon_traces = [trace for trace in figure.data if hasattr(trace, "fill") and trace.fill == "toself"]

    if connect_gaps:
        # connect_gaps=True: Should create exactly ONE polygon trace covering all data points
        assert len(polygon_traces) == 1

        polygon_trace = polygon_traces[0]

        # The single polygon should have 12 points: 6 lower + 6 upper (reversed)
        assert len(polygon_trace.x) == 12
        assert len(polygon_trace.y) == 12

    else:
        # connect_gaps=False: Should create MULTIPLE polygon traces (one per continuous segment)
        # With our gap data, we expect 3 segments: [10:00-11:00], [13:00-14:00], [18:00-19:00]
        assert len(polygon_traces) == 3, (
            f"Expected 3 polygon segments for connect_gaps=False, got {len(polygon_traces)}"
        )

        # Each segment polygon should have 4 points: 2 lower + 2 upper
        for polygon_trace in polygon_traces:
            assert len(polygon_trace.x) == 4
            assert len(polygon_trace.y) == 4

    # Both modes should have a hover trace for quantile information
    hover_traces = [
        trace
        for trace in figure.data
        if hasattr(trace, "showlegend") and not trace.showlegend and getattr(trace, "fill", None) is None
    ]
    assert len(hover_traces) >= 1
