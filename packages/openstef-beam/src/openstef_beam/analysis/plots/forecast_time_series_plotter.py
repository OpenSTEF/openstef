# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Time series plotting for forecast data visualization.

This module provides time series plotting capabilities for comparing
forecasts, measurements, and uncertainty quantiles across multiple models.
"""

from datetime import timedelta
from typing import Any, ClassVar, Self, TypedDict, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseConfig


class ModelData(TypedDict):
    """TypedDict for storing model data."""

    model_name: str
    forecast: pd.Series | None
    quantiles: pd.DataFrame | None


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


class QuantilePolygonStyle(TypedDict):
    """TypedDict for storing quantile polygon styling parameters."""

    fill_color: str
    stroke_color: str
    legendgroup: str


class ForecastTimeSeriesPlotter(BaseConfig):
    """Creates interactive time series charts comparing forecasts, measurements, and uncertainty bands.

    This plotter visualizes forecast performance over time by overlaying multiple models'
    predictions, actual measurements, and uncertainty quantiles on a single chart. The
    resulting interactive plots help answer questions like:

    - How do different models' forecasts compare to actual values over time?
    - Which model provides the most accurate short-term vs long-term predictions?
    - How well do uncertainty bands capture actual forecast errors?
    - Are there seasonal or temporal patterns in model performance?

    The plots include:
    - Line charts for measurements (actual values) and model forecasts
    - Shaded confidence bands showing forecast uncertainty (quantiles)
    - Color-coded models for easy visual comparison
    - Interactive hover information and zooming capabilities

    Example:
        Basic usage comparing forecast to measurements:

        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> import pandas as pd
        >>> from datetime import timedelta
        >>> # Create sample data
        >>> dates = pd.date_range('2024-01-01', periods=10, freq='h')
        >>> measurements = pd.Series(data=range(10), index=dates, name="measurements")
        >>> forecast = pd.Series(data=[x + 1 for x in range(10)], index=dates, name="forecast")
        >>>
        >>> # Create and configure plotter
        >>> plotter = ForecastTimeSeriesPlotter()
        >>> _ = plotter.add_measurements(measurements)
        >>> _ = plotter.add_model("XGBoost", forecast=forecast)
        >>> fig = plotter.plot(title="Energy Forecast Comparison")
        >>> type(fig).__name__
        'Figure'
    """

    MEDIAN_QUANTILE: ClassVar[float] = 50.0

    COLOR_SCHEME: ClassVar[dict[str, str]] = {
        "blue": "Blues",
        "green": "Greens",
        "purple": "Purples",
        "orange": "Oranges",
        "magenta": "Magenta",
        "grey": "Greys",
    }
    colors: ClassVar[list[str]] = list(COLOR_SCHEME.keys())
    colormaps: ClassVar[list[str]] = list(COLOR_SCHEME.values())

    colormap_range: tuple[float, float] = (0.2, 0.8)

    fill_opacity: float = 0.5

    stroke_opacity: float = 0.8
    stroke_width: float = 1.5

    sample_interval: timedelta = Field(
        default=timedelta(minutes=15),
        description="Expected interval between consecutive samples in the time series data.",
    )
    connect_gaps: bool = Field(
        default=True,
        description=(
            "If True, connects data points across missing timestamps with lines. "
            "If False, leaves gaps where data is missing (no interpolation)."
        ),
    )

    _measurements: pd.Series | None = PrivateAttr(default=None)
    _models_data: list[ModelData] = PrivateAttr(default_factory=list[ModelData])
    _limits: list[dict[str, Any]] = PrivateAttr(default_factory=list[dict[str, Any]])

    def _insert_gaps_for_missing_timestamps(self, series: pd.Series, sample_interval: pd.Timedelta) -> pd.Series:
        """Insert NaN values where there are temporal gaps larger than the expected sample interval.

        This ensures that connectgaps=False works properly by creating actual NaN values
        for missing timestamps, rather than just having missing timestamps.

        Args:
            series: The time series data
            sample_interval: Expected interval between consecutive samples

        Returns:
            Series with NaN values inserted at gap locations
        """
        if self.connect_gaps:
            return series

        # Create a complete time index from start to end with the expected frequency
        start_time = cast(pd.Timestamp, series.index[0])
        end_time = cast(pd.Timestamp, series.index[-1])

        # Create complete index with the sample interval
        complete_index = pd.date_range(start=start_time, end=end_time, freq=sample_interval)

        # Reindex the series to the complete index, automatically filling missing values with NaN
        return series.reindex(complete_index)

    def add_measurements(self, measurements: pd.Series) -> Self:
        """Add measurements (realized values) to the plot.

        Args:
            measurements: A Pandas Series containing the realized values.

        Returns:
            ForecastTimeSeriesPlotter: The current instance for method chaining.
        """
        self._measurements = measurements
        return self

    def add_model(
        self,
        model_name: str,
        forecast: pd.Series | None = None,
        quantiles: pd.DataFrame | None = None,
    ) -> "ForecastTimeSeriesPlotter":
        """Add a model's forecast and/or quantile data to the plot.

        Args:
            model_name: The name of the model.
            forecast: A pd.Series containing the forecasted values.
            quantiles: A pd.DataFrame containing quantile data.
                Column names should follow the format 'quantile_P{percentile:02d}',
                e.g., 'quantile_P10', 'quantile_P90'.

        Returns:
            ForecastTimeSeriesPlotter: The current instance for method chaining.

        Raises:
            ValueError: If neither forecast nor quantiles are provided, or if quantile
                column names don't follow the expected format.
        """
        if forecast is None and quantiles is None:
            msg = "At least one of forecast or quantiles must be provided."
            raise ValueError(msg)

        # Validate quantile columns if provided
        if quantiles is not None:
            for col in quantiles.columns:
                if not col.startswith("quantile_P"):
                    msg = f"Column '{col}' does not follow the expected format 'quantile_P{{percentile:02d}}'."
                    raise ValueError(msg)

        model_data: ModelData = {
            "model_name": model_name,
            "forecast": forecast,
            "quantiles": quantiles,
        }

        self._models_data.append(model_data)
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
            name = f"Limit {len(self._limits) + 1}"

        self._limits.append({
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
        """
        # Get colors and legendgroup
        colormap = self.colormaps[model_index % len(self.colormaps)]
        fill_color, stroke_color = self._get_quantile_colors(lower_quantile, colormap)
        legendgroup = f"{model_name}_quantile_{lower_quantile}_{upper_quantile}"

        if self.connect_gaps:
            # Original behavior - create single polygon from raw data
            band = BandData(
                model_name=model_name,
                model_index=model_index,
                lower_quantile=int(lower_quantile),
                upper_quantile=int(upper_quantile),
                lower_data=lower_quantile_data,
                upper_data=upper_quantile_data,
            )
            style = QuantilePolygonStyle(
                fill_color=fill_color,
                stroke_color=stroke_color,
                legendgroup=legendgroup,
            )
            self._add_single_quantile_polygon(figure, band, style)
            # Use raw data for hover
            processed_lower_data = lower_quantile_data
            processed_upper_data = upper_quantile_data
        else:
            # Process data to insert gaps, then create segmented polygons
            if len(lower_quantile_data) > 1:
                estimated_interval = (
                    cast("pd.Series[pd.Timestamp]", lower_quantile_data.index.to_series()).diff().median()
                )
                processed_lower_data = self._insert_gaps_for_missing_timestamps(lower_quantile_data, estimated_interval)
                processed_upper_data = self._insert_gaps_for_missing_timestamps(upper_quantile_data, estimated_interval)
            else:
                processed_lower_data = lower_quantile_data
                processed_upper_data = upper_quantile_data

            # Create segmented polygons for each continuous section
            band = BandData(
                model_name=model_name,
                model_index=model_index,
                lower_quantile=int(lower_quantile),
                upper_quantile=int(upper_quantile),
                lower_data=processed_lower_data,
                upper_data=processed_upper_data,
            )
            style = QuantilePolygonStyle(
                fill_color=fill_color,
                stroke_color=stroke_color,
                legendgroup=legendgroup,
            )
            self._add_segmented_quantile_polygons(figure, band, style)

        # Add hover line for quantile values
        x_data = processed_lower_data.index
        y_lower = processed_lower_data.to_numpy(dtype=np.float32)  # type: ignore
        y_upper = processed_upper_data.to_numpy(dtype=np.float32)  # type: ignore

        figure.add_trace(  # type: ignore[reportUnknownMemberType]
            go.Scatter(
                x=x_data,
                y=y_lower,
                mode="lines",
                line={
                    "width": self.stroke_width,
                    "color": f"rgba{stroke_color[3:-1]}, {self.stroke_opacity})",
                },
                customdata=np.column_stack((y_lower, y_upper)),
                hovertemplate=(
                    f"{lower_quantile}%: %{{customdata[0]:,.4s}}<br>"
                    f"{upper_quantile}%: %{{customdata[1]:,.4s}}"
                    "<extra></extra>"
                ),
                name=f"{model_name} {lower_quantile}%-{upper_quantile}% Hover Info",
                showlegend=False,
                legendgroup=legendgroup,
                connectgaps=self.connect_gaps,
            )
        )

    def _prepare_quantile_bands(self) -> list[BandData]:
        """Prepare quantile band data for plotting.

        Returns:
            List of BandData dictionaries with quantile band information.
        """
        bands: list[BandData] = []
        for model_index, model_data in enumerate(self._models_data):
            if model_data["quantiles"] is None:
                continue

            model_name = model_data["model_name"]
            quantiles = model_data["quantiles"]

            # Extract and sort quantile percentages
            quantile_cols = [col for col in quantiles.columns if col.startswith("quantile_P")]
            percentiles = sorted([int(col.split("P")[1]) for col in quantile_cols])

            # Create band data from widest to narrowest
            for i in range(len(percentiles) // 2):
                lower_quantile, upper_quantile = percentiles[i], percentiles[-(i + 1)]
                if float(lower_quantile) == self.MEDIAN_QUANTILE:
                    continue

                # type: ignore construct a BandData-compatible dict
                bands.append({
                    "model_name": model_name,
                    "model_index": model_index,
                    "lower_quantile": lower_quantile,
                    "upper_quantile": upper_quantile,
                    "lower_data": quantiles[f"quantile_P{lower_quantile:02d}"],
                    "upper_data": quantiles[f"quantile_P{upper_quantile:02d}"],
                })
        return bands

    def _prepare_forecast_lines(self) -> list[LineData]:
        """Prepare forecast line data for plotting.

        Returns:
            List of LineData dictionaries with forecast line information.
        """
        lines: list[LineData] = []
        for model_index, model_data in enumerate(self._models_data):
            model_name = model_data["model_name"]
            forecast = model_data["forecast"]

            if forecast is not None:
                lines.append(
                    LineData(
                        model_name=model_name,
                        model_index=model_index,
                        data=forecast,
                        type="forecast",
                    )
                )
        return lines

    def _prepare_quantile_50th_lines(self) -> list[LineData]:
        """Prepare 50th quantile line data for plotting (only when no forecast is provided).

        Returns:
            List of LineData dictionaries with 50th quantile line information.
        """
        lines: list[LineData] = []
        for model_index, model_data in enumerate(self._models_data):
            model_name = model_data["model_name"]
            quantiles = model_data["quantiles"]
            forecast = model_data["forecast"]

            if quantiles is not None and forecast is None:
                median_col = f"quantile_P{int(self.MEDIAN_QUANTILE):02d}"
                if median_col in quantiles.columns:
                    lines.append(
                        LineData(
                            model_name=model_name,
                            model_index=model_index,
                            data=quantiles[median_col],
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

            if self.connect_gaps or len(line["data"]) == 1:
                # Use data as-is (either connect_gaps=True or single data point)
                x_data = line["data"].index
                y_data = line["data"]
            else:
                # Process data to insert gaps for missing timestamps
                estimated_interval = cast("pd.Series[pd.Timestamp]", line["data"].index.to_series()).diff().median()
                processed_data = self._insert_gaps_for_missing_timestamps(line["data"], estimated_interval)
                x_data = processed_data.index
                y_data = processed_data

            figure.add_trace(  # type: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    line={"color": color, "width": self.stroke_width},
                    name=name,
                    customdata=y_data.values,
                    hovertemplate=f"{hover_label}: %{{customdata:,.4s}}<extra></extra>",
                    connectgaps=self.connect_gaps,
                )
            )

    def _add_measurements_to_figure(self, figure: go.Figure) -> None:
        """Add measurements to the figure."""
        if self._measurements is not None:
            if self.connect_gaps:
                # Original behavior - use data as-is
                measurements_data = self._measurements
                x_data = measurements_data.index
                y_data = measurements_data
            else:
                # Process data to insert gaps for missing timestamps
                measurements_data = self._measurements
                processed_data = self._insert_gaps_for_missing_timestamps(
                    measurements_data, pd.Timedelta(self.sample_interval)
                )
                x_data = processed_data.index
                y_data = processed_data

            figure.add_trace(  # type: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    line={"color": "red", "width": self.stroke_width},
                    customdata=y_data.values,
                    hovertemplate="Realized: %{customdata:,.4s}<extra></extra>",
                    name="Realized",
                    connectgaps=self.connect_gaps,
                )
            )

    def _add_limits_to_figure(self, figure: go.Figure) -> None:
        """Add horizontal limit lines to the figure."""
        for limit in self._limits:
            figure.add_hline(  # type: ignore[reportUnknownMemberType]
                y=limit["value"],
                line_dash="dot",
                annotation_text=limit["name"],
                annotation_position="top right",
            )

    @staticmethod
    def _configure_figure_layout(figure: go.Figure, title: str) -> None:
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

        Raises:
            ValueError: If no data has been added to the plotter.
        """
        if not self._models_data and self._measurements is None:
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
        ForecastTimeSeriesPlotter._configure_figure_layout(figure, title)

        return figure

    def _add_single_quantile_polygon(
        self,
        figure: go.Figure,
        band: BandData,
        style: QuantilePolygonStyle,
    ) -> None:
        """Add a single quantile polygon for the original connect_gaps=True behavior."""
        # Create polygon shape for the quantile band in counterclockwise order
        index_list = band["lower_data"].index.to_list()
        x = index_list + index_list[::-1]
        y = list(band["lower_data"]) + list(band["upper_data"][::-1])

        figure.add_trace(  # type: ignore[reportUnknownMemberType]
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=f"rgba{style['fill_color'][3:-1]}, {self.fill_opacity})",
                line={
                    "color": f"rgba{style['stroke_color'][3:-1]}, {self.stroke_opacity})",
                    "width": self.stroke_width,
                },
                name=f"{band['model_name']} {band['lower_quantile']}%-{band['upper_quantile']}%",
                showlegend=True,
                hoverinfo="skip",
                legendgroup=style["legendgroup"],
            )
        )

    def _add_segmented_quantile_polygons(
        self,
        figure: go.Figure,
        band: BandData,
        style: QuantilePolygonStyle,
    ) -> None:
        """Add segmented quantile polygons that respect gaps (NaN values)."""
        # Find continuous segments (non-NaN data)
        mask = ~(band["lower_data"].isna() | band["upper_data"].isna())

        if not mask.any():
            return  # No valid data

        # Find start and end indices of continuous segments
        # diff() finds transitions: True->False (-1) and False->True (1)
        mask_array = mask.to_numpy()  # type: ignore
        transitions = np.diff(np.concatenate(([False], mask_array, [False])).astype(int))
        segment_starts = np.where(transitions == 1)[0]  # Where transitions from False to True
        segment_ends = np.where(transitions == -1)[0] - 1  # Where transitions from True to False
        segments = list(zip(segment_starts, segment_ends, strict=True))

        # Create separate polygon for each continuous segment
        for seg_idx, (start, end) in enumerate(segments):  # type: ignore[misc]
            lower_segment = band["lower_data"].iloc[start : end + 1]
            upper_segment = band["upper_data"].iloc[start : end + 1]

            # Create polygon for this segment
            index_list = lower_segment.index.to_list()
            x = index_list + index_list[::-1]
            y = list(lower_segment) + list(upper_segment[::-1])

            # Only show legend for the first segment
            show_legend = seg_idx == 0

            figure.add_trace(  # type: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor=f"rgba{style['fill_color'][3:-1]}, {self.fill_opacity})",
                    line={
                        "color": f"rgba{style['stroke_color'][3:-1]}, {self.stroke_opacity})",
                        "width": self.stroke_width,
                    },
                    name=f"{band['model_name']} {band['lower_quantile']}%-{band['upper_quantile']}%",
                    showlegend=show_legend,
                    hoverinfo="skip",
                    legendgroup=style["legendgroup"],
                )
            )


__all__ = ["ForecastTimeSeriesPlotter"]
