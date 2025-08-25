# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import cast

import pandas as pd
import plotly.graph_objects as go
import pytest

from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
from openstef_core.datasets import TimeSeriesDataset


def test_add_model_with_forecast_only():
    # Arrange
    forecast = TimeSeriesDataset(
        data=pd.Series(data=[1, 2, 3], index=pd.date_range("2023-01-01", periods=3, freq="D")).to_frame(
            name="measurement"
        ),
        sample_interval=pd.Timedelta("1D"),
    )

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model A", forecast=forecast)

    # Assert
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model_name"] == "Model A"
    assert plotter.models_data[0]["forecast"] == forecast
    assert plotter.models_data[0]["quantiles"] is None


def test_add_model_with_quantiles_only():
    # Arrange
    quantiles = TimeSeriesDataset(
        data=pd.DataFrame(
            data={"quantile_P10": [1, 2], "quantile_P90": [3, 4]},
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        ),
        sample_interval=pd.Timedelta("1D"),
    )

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model B", quantiles=quantiles)

    # Assert
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model_name"] == "Model B"
    assert plotter.models_data[0]["forecast"] is None
    assert plotter.models_data[0]["quantiles"] == quantiles


def test_add_model_with_forecast_and_quantiles():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="D")

    forecast = TimeSeriesDataset(
        data=pd.Series(
            data=[1, 2, 3],
            index=index,
        ).to_frame(name="measurement"),
        sample_interval=pd.Timedelta("1D"),
    )
    quantiles = TimeSeriesDataset(
        data=pd.DataFrame(data={"quantile_P10": [1, 2, 3], "quantile_P90": [4, 5, 6]}, index=index),
        sample_interval=pd.Timedelta("1D"),
    )

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model C", forecast=forecast, quantiles=quantiles)

    # Assert
    assert len(plotter.models_data) == 1
    assert plotter.models_data[0]["model_name"] == "Model C"
    assert plotter.models_data[0]["forecast"] == forecast
    assert plotter.models_data[0]["quantiles"] == quantiles


def test_method_chaining():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    measurements = TimeSeriesDataset(
        data=pd.Series([10, 20, 30], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )
    forecast1 = TimeSeriesDataset(
        data=pd.Series([1, 2, 3], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )
    forecast2 = TimeSeriesDataset(
        data=pd.Series([4, 5, 6], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )

    # Act - test method chaining
    plotter = (
        ForecastTimeSeriesPlotter()
        .add_measurements(measurements)
        .add_model(model_name="Model 1", forecast=forecast1)
        .add_model(model_name="Model 2", forecast=forecast2)
    )

    # Assert
    assert plotter.measurements is not None
    assert plotter.measurements == measurements
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
    quantiles = TimeSeriesDataset(
        data=pd.DataFrame(
            {"bad_column": [1, 2], "quantile_P90": [3, 4]}, index=pd.date_range("2023-01-01", periods=2, freq="D")
        ),
        sample_interval=pd.Timedelta("1D"),
    )
    plotter = ForecastTimeSeriesPlotter()

    # Act & Assert
    with pytest.raises(ValueError, match="Column 'bad_column' does not follow the expected format"):
        plotter.add_model(model_name="Model Y", quantiles=quantiles)


def test_add_measurements():
    # Arrange
    measurements = TimeSeriesDataset(
        data=pd.Series(data=[10, 20, 30], index=pd.date_range("2024-01-01", periods=3, freq="D")).to_frame(
            name="measurement"
        ),
        sample_interval=pd.Timedelta("1D"),
    )

    # Act
    plotter = ForecastTimeSeriesPlotter()
    plotter.add_measurements(measurements)

    # Assert
    assert plotter.measurements is not None
    assert plotter.measurements == measurements


def test_plot_with_no_data_raises():
    # Arrange
    plotter = ForecastTimeSeriesPlotter()

    # Act & Assert
    with pytest.raises(ValueError, match=r"No data has been added. Use add_measurements or add_model first."):
        plotter.plot()


def test_plot_with_only_measurements():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    measurements = TimeSeriesDataset(
        data=pd.Series([10, 20, 30], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )

    plotter = ForecastTimeSeriesPlotter()
    plotter.add_measurements(measurements)

    # Act
    fig = plotter.plot()

    # Assert
    assert isinstance(fig, go.Figure)
    # Should have one trace for the measurements
    assert len(fig.data) == 1


def test_plot_with_forecasts_and_no_quantiles():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    forecast1 = TimeSeriesDataset(
        data=pd.Series([1, 2, 3], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )
    forecast2 = TimeSeriesDataset(
        data=pd.Series([4, 5, 6], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )

    plotter = (
        ForecastTimeSeriesPlotter()
        .add_model(model_name="Model 1", forecast=forecast1)
        .add_model(model_name="Model 2", forecast=forecast2)
    )

    # Act
    fig = plotter.plot()

    # Assert
    # Should have two traces, one for each forecast
    assert len(fig.data) == 2


def test_plot_with_measurements_and_forecasts():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    measurements = TimeSeriesDataset(
        data=pd.Series([10, 20, 30], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )
    forecast = TimeSeriesDataset(
        data=pd.Series([1, 2, 3], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )

    plotter = (
        ForecastTimeSeriesPlotter().add_measurements(measurements).add_model(model_name="Model 1", forecast=forecast)
    )

    # Act
    fig = plotter.plot()

    # Assert
    # Should have two traces: measurements and forecast
    assert len(fig.data) == 2


def test_plot_with_forecasts_and_quantiles():
    # Arrange
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    forecast = TimeSeriesDataset(
        data=pd.Series([1, 2, 3], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )
    quantiles = TimeSeriesDataset(
        data=pd.DataFrame(
            {"quantile_P10": [0, 1, 2], "quantile_P50": [1, 2, 3], "quantile_P90": [2, 3, 4]}, index=index
        ),
        sample_interval=pd.Timedelta("1D"),
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
    measurements = TimeSeriesDataset(
        data=pd.Series([10, 20, 30], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )

    forecast1 = TimeSeriesDataset(
        data=pd.Series([1, 2, 3], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )

    quantiles1 = TimeSeriesDataset(
        data=pd.DataFrame(
            {"quantile_P10": [0, 1, 2], "quantile_P50": [1, 2, 3], "quantile_P90": [2, 3, 4]}, index=index
        ),
        sample_interval=pd.Timedelta("1D"),
    )

    forecast2 = TimeSeriesDataset(
        data=pd.Series([4, 5, 6], index=index).to_frame(name="value"),
        sample_interval=pd.Timedelta("1D"),
    )
    quantiles2 = TimeSeriesDataset(
        data=pd.DataFrame(
            {"quantile_P10": [3, 4, 5], "quantile_P50": [4, 5, 6], "quantile_P90": [5, 6, 7]}, index=index
        ),
        sample_interval=pd.Timedelta("1D"),
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
    assert len(fig.data) == 7


def test_custom_title():
    # Arrange
    measurements = TimeSeriesDataset(
        data=pd.Series(data=[10, 20, 30], index=pd.date_range("2024-01-01", periods=3, freq="D")).to_frame(
            name="measurement"
        ),
        sample_interval=pd.Timedelta("1D"),
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
    quantiles = TimeSeriesDataset(
        data=pd.DataFrame(
            {"quantile_P10": [0, 1, 2], "quantile_P50": [1, 2, 3], "quantile_P90": [2, 3, 4]}, index=index
        ),
        sample_interval=pd.Timedelta("1D"),
    )

    plotter = ForecastTimeSeriesPlotter()
    plotter.add_model(model_name="Model 1", quantiles=quantiles)

    # Act
    fig = plotter.plot()

    # Assert
    # Should have quantile bands (2 traces for 10/90) plus 50th quantile line (1 trace): total 3
    assert len(fig.data) == 3
    names: list[str] = [trace.name for trace in fig.data]
    assert any("Quantile (50th)" in n for n in names)
    assert any("10%-90%" in n for n in names)
