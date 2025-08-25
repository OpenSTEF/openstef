# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Any, ClassVar, TypedDict, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from openstef_core.datasets import TimeSeriesDataset


class ModelData(TypedDict):
    """TypedDict for storing model data."""

    model_name: str
    forecast: TimeSeriesDataset | None
    quantiles: TimeSeriesDataset | None


class LineData(TypedDict):
    """TypedDict for storing line data."""

    model_name: str
    model_index: int
    data: pd.Series
    type: str  # "forecast" or "quantile_50th"


# Add a typed dict for quantile band data
class BandData(TypedDict):
    """TypedDict for storing quantile band data."""

    model_name: str
    model_index: int
    lower_quantile: int
    upper_quantile: int
    lower_data: pd.Series
    upper_data: pd.Series


class ForecastTimeSeriesPlotter:
    """A class for plotting time series data for measurements, forecasts and quantiles of multiple models."""

    MEDIAN_QUANTILE: ClassVar[float] = 50.0

    COLOR_SCHEME: ClassVar[dict[str, str]] = {
        "blue": "Blues",
        "green": "Greens",
        "purple": "Purples",
        "orange": "Oranges",
    }
    colors: ClassVar[list[str]] = list(COLOR_SCHEME.keys())
    colormaps: ClassVar[list[str]] = list(COLOR_SCHEME.values())

    colormap_range: tuple[float, float] = (0.2, 0.8)

    fill_opacity: float = 0.5

    stroke_opacity: float = 0.8
    stroke_width: float = 1.5

    def __init__(self):
        """Initialize the ForecastTimeSeriesPlotter."""
        self.measurements: TimeSeriesDataset | None = None
        self.models_data: list[ModelData] = []
        self.limits: list[dict[str, Any]] = []

    def add_measurements(self, measurements: TimeSeriesDataset) -> "ForecastTimeSeriesPlotter":
        """Add measurements (realized values) to the plot.

        Args:
            measurements (TimeSeriesDataset): A TimeSeriesDataset containing the realized values.

        Returns:
            ForecastTimeSeriesPlotter: The current instance for method chaining.
        """
        if len(measurements.feature_names) > 1:
            msg = "Measurements dataset must contain exactly one feature/column."
            raise ValueError(msg)

        self.measurements = measurements
        return self

    def add_model(
        self,
        model_name: str,
        forecast: TimeSeriesDataset | None = None,
        quantiles: TimeSeriesDataset | None = None,
    ) -> "ForecastTimeSeriesPlotter":
        """Add a model's forecast and/or quantile data to the plot.

        Args:
            model_name (str): The name of the model.
            forecast (TimeSeriesDataset | None): A TimeSeriesDataset containing the forecasted values.
            quantiles (TimeSeriesDataset | None): A TimeSeriesDataset containing quantile data.
                Column names should follow the format 'quantile_P{percentile:02d}',
                e.g., 'quantile_P10', 'quantile_P90'.

        Returns:
            ForecastTimeSeriesPlotter: The current instance for method chaining.
        """
        if forecast is None and quantiles is None:
            msg = "At least one of forecast or quantiles must be provided."
            raise ValueError(msg)

        # Validate quantile columns if provided
        if quantiles is not None:
            for col in quantiles.feature_names:
                if not col.startswith("quantile_P"):
                    msg = f"Column '{col}' does not follow the expected format 'quantile_P{{percentile:02d}}'."
                    raise ValueError(msg)

        model_data: ModelData = {
            "model_name": model_name,
            "forecast": forecast,
            "quantiles": quantiles,
        }

        self.models_data.append(model_data)
        return self

    def add_limit(
        self,
        value: float,
        name: str | None = None,
    ) -> "ForecastTimeSeriesPlotter":
        """Add a horizontal limit line to the plot.

        Args:
            value (float): The y-value where the limit line should be drawn.
            name (str, optional): The name of the limit to be displayed in the legend.
                If not provided, will be named "Limit N" where N is the index.

        Returns:
            ForecastTimeSeriesPlotter: The current instance for method chaining.
        """
        if name is None:
            name = f"Limit {len(self.limits) + 1}"

        self.limits.append({
            "value": value,
            "name": name,
        })

        return self

    def _get_color_by_value(self, value: float, colormap: str) -> str:
        """Maps a normalized value to a color using the specified colormap.

        Args:
            value (float): A value between 0 and 1 to be mapped to a color.
            colormap (str): The colormap to use.

        Returns:
            str: A color in the rgba format.
        """
        rescaled = self.colormap_range[0] + value * (self.colormap_range[1] - self.colormap_range[0])
        rescaled = min(1.0, max(0.0, rescaled))

        color_result = px.colors.sample_colorscale(colorscale=colormap, samplepoints=[rescaled])[0]  # type: ignore[reportUnknownMemberType]
        return str(color_result)  # type: ignore[arg-type]

    def _get_quantile_colors(self, quantile: float, colormap: str) -> tuple[str, str]:
        """Generate fill and stroke colors for a given quantile using a colorscale.

        Colors are determined based on the distance from the median (50th percentile).

        Args:
            quantile (float): The quantile value (0-100) to generate colors for.
            colormap (str): The colorscale to use.

        Returns:
            Tuple[str, str]: A tuple containing (fill_color, stroke_color).
        """
        fill_value = 1 - abs(quantile - 50.0) / 50.0
        stroke_value = 1 - abs(quantile + 5.0 - 50.0) / 50.0
        return (
            self._get_color_by_value(fill_value, colormap),
            self._get_color_by_value(stroke_value, colormap),
        )

    def _add_quantile_band(
        self,
        figure: go.Figure,
        lower_quantile_data: pd.Series,
        lower_quantile: float,
        upper_quantile_data: pd.Series,
        upper_quantile: float,
        model_name: str,
        model_index: int,
    ):
        """Add a quantile band to the plotly figure.

        Creates a filled polygon representing the area between lower and upper quantiles,
        and adds it to the provided figure along with hover information.

        Args:
            figure (go.Figure): The plotly figure to add the quantile band to.
            lower_quantile_data (pd.Series): Series with data for the lower quantile.
            lower_quantile (float): The percentile value of the lower quantile.
            upper_quantile_data (pd.Series): Series with data for the upper quantile.
            upper_quantile (float): The percentile value of the upper quantile.
            model_name (str): The name of the model for which the quantile band is being added.
            model_index (int): The index of the model in the models_data list.

        Returns:
            None: The figure is modified in place.
        """
        # Create polygon shape for the quantile band in counterclockwise order
        lower_quantile_index_list = lower_quantile_data.index.to_list()
        upper_quantile_index_list = upper_quantile_data.index.to_list()
        x = lower_quantile_index_list + upper_quantile_index_list[::-1]
        y = list(lower_quantile_data) + list(upper_quantile_data[::-1])

        # Get colors for the band
        colormap = self.colormaps[model_index % len(self.colormaps)]
        fill_color, stroke_color = self._get_quantile_colors(lower_quantile, colormap)

        # Group traces by quantile range
        legendgroup = f"{model_name}_quantile_{lower_quantile}_{upper_quantile}"

        # Add a single trace that forms a filled polygon
        figure.add_trace(  # type: ignore[reportUnknownMemberType]
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                fillcolor=f"rgba{fill_color[3:-1]}, {self.fill_opacity})",
                line={
                    "color": f"rgba{stroke_color[3:-1]}, {self.stroke_opacity})",
                    "width": self.stroke_width,
                },
                name=f"{model_name} {lower_quantile}%-{upper_quantile}%",
                showlegend=True,
                hoverinfo="skip",
                legendgroup=legendgroup,
            )
        )

        # Add an (invisible) line around the filled area to make quantile
        # values selectable/hover-able.
        # Hovering on filled area values is not supported by plotly.
        figure.add_trace(  # type: ignore[reportUnknownMemberType]
            go.Scatter(
                x=lower_quantile_data.index,
                y=lower_quantile_data.to_numpy(),
                mode="lines",
                line={
                    "width": self.stroke_width,
                    "color": f"rgba{stroke_color[3:-1]}, {self.stroke_opacity})",
                },
                customdata=np.column_stack((lower_quantile_data.to_numpy(), upper_quantile_data.to_numpy())),
                hovertemplate=(
                    f"{lower_quantile}%: %{{customdata[0]:,.4s}}<br>"
                    f"{upper_quantile}%: %{{customdata[1]:,.4s}}"
                    "<extra></extra>"
                ),
                name=f"{model_name} {lower_quantile}%-{upper_quantile}% Hover Info",
                showlegend=False,
                legendgroup=legendgroup,
            )
        )

    def _prepare_quantile_bands(self) -> list[BandData]:
        """Prepare quantile band data for plotting."""
        bands: list[BandData] = []
        for model_index, model_data in enumerate(self.models_data):
            if model_data["quantiles"] is None:
                continue

            model_name = model_data["model_name"]
            quantiles = model_data["quantiles"]

            # Extract and sort quantile percentages
            quantile_cols = [col for col in quantiles.feature_names if col.startswith("quantile_P")]
            percentiles = sorted([int(col.split("P")[1]) for col in quantile_cols])

            # Create band data from widest to narrowest
            for i in range(len(percentiles) // 2):
                lower_quantile, upper_quantile = percentiles[i], percentiles[-(i + 1)]
                if float(lower_quantile) == self.MEDIAN_QUANTILE:
                    continue

                # type: ignore construct a BandData-compatible dict
                bands.append(
                    {
                        "model_name": model_name,
                        "model_index": model_index,
                        "lower_quantile": lower_quantile,
                        "upper_quantile": upper_quantile,
                        "lower_data": quantiles.data[f"quantile_P{lower_quantile:02d}"],
                        "upper_data": quantiles.data[f"quantile_P{upper_quantile:02d}"],
                    }  # type: ignore[typeddict-item]
                )
        return bands

    def _prepare_forecast_lines(self) -> list[LineData]:
        """Prepare forecast line data for plotting."""
        lines: list[LineData] = []
        for model_index, model_data in enumerate(self.models_data):
            model_name = model_data["model_name"]
            forecast = model_data["forecast"]

            if forecast is not None:
                lines.append(
                    LineData(
                        model_name=model_name,
                        model_index=model_index,
                        data=cast(pd.Series, forecast.data.squeeze()),
                        type="forecast",
                    )
                )
        return lines

    def _prepare_quantile_50th_lines(self) -> list[LineData]:
        """Prepare 50th quantile line data for plotting (only when no forecast is provided)."""
        lines: list[LineData] = []
        for model_index, model_data in enumerate(self.models_data):
            model_name = model_data["model_name"]
            quantiles = model_data["quantiles"]
            forecast = model_data["forecast"]

            if quantiles is not None and forecast is None:
                median_col = f"quantile_P{int(self.MEDIAN_QUANTILE):02d}"
                if median_col in quantiles.feature_names:
                    lines.append(
                        LineData(
                            model_name=model_name,
                            model_index=model_index,
                            data=quantiles.data[median_col],
                            type="quantile_50th",
                        )
                    )
        return lines

    def _add_quantile_bands_to_figure(self, figure: go.Figure, bands: list[BandData]) -> None:
        """Add quantile bands to the figure."""
        for band in bands:
            self._add_quantile_band(
                figure=figure,
                lower_quantile_data=band["lower_data"],
                lower_quantile=band["lower_quantile"],
                upper_quantile_data=band["upper_data"],
                upper_quantile=band["upper_quantile"],
                model_name=band["model_name"],
                model_index=band["model_index"],
            )

    def _add_lines_to_figure(self, figure: go.Figure, lines: list[LineData]) -> None:
        """Add forecast and quantile lines to the figure."""
        for line in lines:
            color: str = self.colors[line["model_index"] % len(self.colors)]  # type: ignore[assignment]

            if line["type"] == "forecast":
                name = f"{line['model_name']} Forecast (50th)"
                hover_label = "Forecast (50th)"
            else:  # quantile_50th
                name = f"{line['model_name']} Quantile (50th)"
                hover_label = "Quantile (50th)"

            figure.add_trace(  # type: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=line["data"].index,
                    y=line["data"],
                    mode="lines",
                    line={"color": color, "width": self.stroke_width},
                    name=name,
                    customdata=line["data"].values,
                    hovertemplate=f"{hover_label}: %{{customdata:,.4s}}<extra></extra>",
                )
            )

    def _add_measurements_to_figure(self, figure: go.Figure) -> None:
        """Add measurements to the figure."""
        if self.measurements is not None:
            figure.add_trace(  # type: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=self.measurements.index,
                    y=self.measurements.data,
                    mode="lines",
                    line={"color": "red", "width": self.stroke_width},
                    customdata=self.measurements.data.values,
                    hovertemplate="Realized: %{customdata:,.4s}<extra></extra>",
                    name="Realized",
                )
            )

    def _add_limits_to_figure(self, figure: go.Figure) -> None:
        """Add horizontal limit lines to the figure."""
        for limit in self.limits:
            figure.add_hline(  # type: ignore[reportUnknownMemberType]
                y=limit["value"],
                line_dash="dot",
                annotation_text=limit["name"],
                annotation_position="top right",
            )

    def _configure_figure_layout(self, figure: go.Figure, title: str) -> None:
        """Configure the figure layout and styling."""
        figure.update_layout(  # type: ignore[reportUnknownMemberType]
            title=title,
            xaxis_title="Datetime [UTC]",
            yaxis_title="Load [W]",
            template="plotly_white",
            hovermode="x unified",
        )

    def plot(self, title: str = "Time Series Plots") -> go.Figure:
        """Plot the time series data with measurements, forecasts, and quantiles.

        Args:
            title (str): Title of the plot.

        Returns:
            plotly.graph_objects.Figure: The time series plot.
        """
        if not self.models_data and self.measurements is None:
            msg = "No data has been added. Use add_measurements or add_model first."
            raise ValueError(msg)

        # Prepare data for plotting
        quantile_bands: list[BandData] = self._prepare_quantile_bands()
        forecast_lines: list[LineData] = self._prepare_forecast_lines()
        quantile_50th_lines: list[LineData] = self._prepare_quantile_50th_lines()

        # Create figure and add traces in the correct order
        figure = go.Figure()

        # Layer 1: Quantile bands (background)
        self._add_quantile_bands_to_figure(figure, quantile_bands)

        # Layer 2: Forecast lines and 50th quantile lines (middle)
        all_lines = forecast_lines + quantile_50th_lines
        self._add_lines_to_figure(figure, all_lines)

        # Layer 3: Measurements (foreground)
        self._add_measurements_to_figure(figure)

        # Add limits and configure layout
        self._add_limits_to_figure(figure)
        self._configure_figure_layout(figure, title)

        return figure


__all__ = ["ForecastTimeSeriesPlotter"]
