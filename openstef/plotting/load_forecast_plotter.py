# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Optional

import numpy as np
from pydantic import BaseModel
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class LoadForecastPlotter(BaseModel):
    colormap: str = "Blues"
    colormap_range: Tuple[float, float] = (0.2, 0.8)

    fill_opacity: float = 0.5
    stroke_opacity: float = 0.8
    stroke_width: float = 1.5

    def _get_color_by_value(self, value: float) -> str:
        """Maps a normalized value to a color using the specified colormap.

        Args:
            value (float): A value between 0 and 1 to be mapped to a color.

        Returns:
            str: A color in the rgba format.
        """
        rescaled = self.colormap_range[0] + value * (
            self.colormap_range[1] - self.colormap_range[0]
        )
        rescaled = min(1.0, max(0.0, rescaled))

        return px.colors.sample_colorscale(
            colorscale=self.colormap, samplepoints=[rescaled]
        )[0]

    def _get_quantile_colors(self, quantile: float) -> Tuple[str, str]:
        """Generate fill and stroke colors for a given quantile using a colorscale.

        Colors are determined based on the distance from the median (50th percentile).

        Args:
           quantile (float): The quantile value (0-100) to generate colors for.

        Returns:
           Tuple[str, str]: A tuple containing (fill_color, stroke_color).
        """
        fill_value = 1 - abs(quantile - 50.0) / 50.0
        stroke_value = 1 - abs(quantile + 5.0 - 50.0) / 50.0
        return (
            self._get_color_by_value(fill_value),
            self._get_color_by_value(stroke_value),
        )

    def _add_quantile_band(
        self,
        figure: go.Figure,
        lower_quantile_data: pd.Series,
        lower_quantile: float,
        upper_quantile_data: pd.Series,
        upper_quantile: float,
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

        Returns:
            None: The figure is modified in place.
        """
        # Create polygon shape for the quantile band in counterclockwise order
        x = list(lower_quantile_data.index) + list(upper_quantile_data.index[::-1])
        y = list(lower_quantile_data) + list(upper_quantile_data[::-1])

        # Get colors for the band
        fill_color, stroke_color = self._get_quantile_colors(lower_quantile)

        # Group traces by quantile range
        legendgroup = f"quantile_{lower_quantile}_{upper_quantile}"

        # Add a single trace that forms a filled polygon
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                fillcolor=f"rgba{fill_color[3:-1]}, {self.fill_opacity})",
                line=dict(
                    color=f"rgba{stroke_color[3:-1]}, {self.stroke_opacity})",
                    width=self.stroke_width,
                ),
                name=f"{lower_quantile}%-{upper_quantile}%",
                showlegend=True,
                hoverinfo="skip",
                legendgroup=legendgroup,
            )
        )

        # Add an (invisible) line around the filled area to make quantile
        # values selectable/hover-able.
        # Hovering on filled area values is not supported by plotly.
        figure.add_trace(
            go.Scatter(
                x=lower_quantile_data.index,
                y=lower_quantile_data.values,
                mode="lines",
                line=dict(
                    width=self.stroke_width,
                    color=f"rgba{stroke_color[3:-1]}, {self.stroke_opacity})",
                ),
                customdata=np.column_stack(
                    (lower_quantile_data.values, upper_quantile_data.values)
                ),
                hovertemplate=(
                    f"{lower_quantile}%: %{{customdata[0]:,.4s}}<br>"
                    f"{upper_quantile}%: %{{customdata[1]:,.4s}}"
                    "<extra></extra>"
                ),
                name=f"{lower_quantile}%-{upper_quantile}% Hover Info",
                showlegend=False,
                legendgroup=legendgroup,
            )
        )

    def plot(
        self,
        realized: Optional[pd.Series] = None,
        forecast: Optional[pd.Series] = None,
        quantiles: Optional[pd.DataFrame] = None,
    ):
        """Create a plot showing forecast quantiles and realized values.

        Generates an interactive plotly figure displaying the forecast distribution
        through quantile bands, the median forecast, and the actual realized values.

        Args:
            realized (pd.Series): Time series of realized (actual) values.
            forecast (pd.Series): Time series of forecast values (typically the median).
            quantiles (pd.DataFrame): DataFrame containing quantile predictions.
                Column names should follow the format 'quantile_P{percentile:02d}',
                e.g., 'quantile_P10', 'quantile_P90'.

        Returns:
            go.Figure: A plotly figure object with the configured visualization.
        """
        figure = go.Figure()

        if quantiles is not None:
            # Extract and sort quantile percentages
            quantile_cols = [
                col for col in quantiles.columns if col.startswith("quantile_P")
            ]
            percentiles = sorted([int(col.split("P")[1]) for col in quantile_cols])

            # Add quantile bands from widest to narrowest
            for i in range(len(percentiles) // 2):
                lower_quantile, upper_quantile = percentiles[i], percentiles[-(i + 1)]
                if float(lower_quantile) == 50.0:
                    continue

                self._add_quantile_band(
                    figure=figure,
                    lower_quantile_data=quantiles[f"quantile_P{lower_quantile:02d}"],
                    lower_quantile=lower_quantile,
                    upper_quantile_data=quantiles[f"quantile_P{upper_quantile:02d}"],
                    upper_quantile=upper_quantile,
                )

        if forecast is not None:
            # Add forecast line (50th percentile)
            figure.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=forecast,
                    mode="lines",
                    line=dict(color="blue", width=self.stroke_width),
                    name="Forecast (50th)",
                    customdata=forecast.values,
                    hovertemplate="Forecast (50th): %{customdata:,.4s}<extra></extra>",
                )
            )

        if realized is not None:
            # Add realized values on top
            figure.add_trace(
                go.Scatter(
                    x=realized.index,
                    y=realized,
                    mode="lines",
                    line=dict(color="red", width=self.stroke_width),
                    customdata=realized.values,
                    hovertemplate="Realized: %{customdata:,.4s}<extra></extra>",
                    name="Realized",
                )
            )

        # Styling configuration
        figure.update_layout(
            title=f"Load Forecast vs Actual",
            xaxis_title="Datetime [UTC]",
            yaxis_title="Load [W]",
            template="plotly_white",
            hovermode="x unified",
        )

        return figure
