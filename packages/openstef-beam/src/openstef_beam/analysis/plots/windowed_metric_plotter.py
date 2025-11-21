# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Windowed metric plotting for time-based performance analysis.

This module provides visualization tools for analyzing model performance
over time using rolling windows, enabling trend analysis and temporal
pattern identification in forecast accuracy.
"""

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class WindowedMetricPlotter:
    """Creates time series plots of performance metrics using windowed analysis.

    This plotter visualizes how model performance changes over time by computing
    metrics within rolling time windows. This helps identify temporal patterns,
    seasonal variations, and long-term trends in forecast accuracy.

    The plots show:
    - Performance metrics over time for each model
    - Rolling averages to smooth out short-term fluctuations
    - Comparative analysis across multiple models
    - Trend identification for performance monitoring

    Example:
        Basic usage for RMSE over time:

        >>> from datetime import datetime
        >>> plotter = WindowedMetricPlotter()
        >>> timestamps = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        >>> rmse_values = [0.12, 0.15]
        >>> _ = plotter.add_model("XGBoost", timestamps, rmse_values)
        >>> _ = plotter.set_window_size("1D")
        >>> fig = plotter.plot(title="RMSE Over Time", metric_name="RMSE")
        >>> type(fig).__name__
        'Figure'
    """

    def __init__(self):
        """Initialize the plotter with empty data containers."""
        # Models data contains the model name, timestamps, and metric values
        self.models_data: list[dict[str, str | list[datetime] | list[float]]] = []
        self.window_size: str | None = None

    def add_model(
        self,
        model_name: str,
        timestamps: list[datetime],
        metric_values: list[float],
    ) -> "WindowedMetricPlotter":
        """Add a model's metric values and timestamps to the plot.

        Args:
            model_name (str): The name of the model.
            timestamps (List[datetime]): List of datetime objects for the x-axis.
            metric_values (List[float]): List of metric values for the y-axis.

        Returns:
            WindowedMetricPlotter: The current instance for method chaining.

        Raises:
            ValueError: If timestamps and metric values have different lengths.
        """
        if len(timestamps) != len(metric_values):
            msg = "Timestamps and metric values must have the same length"
            raise ValueError(msg)

        model_data = {
            "model": model_name,
            "timestamp": timestamps,
            "metric_value": metric_values,
        }

        self.models_data.append(model_data)
        return self

    def set_window_size(self, window_size: str) -> "WindowedMetricPlotter":
        """Set the window size used for metric calculation.

        Args:
            window_size (str): Description of the window size (e.g., "7 days").

        Returns:
            WindowedMetricPlotter: The current instance for method chaining.
        """
        self.window_size = window_size
        return self

    def plot(self, title: str = "Metric over time", metric_name: str = "Metric") -> go.Figure:
        """Create and return a line chart of metrics over time for all added models.

        Args:
            title (str): Main title of the plot.
            metric_name (str): Name of the metric being plotted.

        Returns:
            plotly.graph_objects.Figure: The resulting plot.

        Raises:
            ValueError: If no model data has been added.
        """
        if not self.models_data:
            msg = "No model data has been added. Use add_model first."
            raise ValueError(msg)

        # Combine all model data into a single DataFrame
        model_df_list: list[pd.DataFrame] = []
        for model_data in self.models_data:
            model_df = pd.DataFrame({
                "model": [model_data["model"]] * len(model_data["timestamp"]),
                "timestamp": model_data["timestamp"],
                "metric_value": model_data["metric_value"],
            })
            model_df_list.append(model_df)

        models_df = pd.concat(model_df_list, ignore_index=True)

        # Add subtitle to title if window size is provided
        if self.window_size:
            title += f"<br><sup>Window size: {self.window_size}</sup>"

        fig: go.Figure = px.line(models_df, x="timestamp", y="metric_value", color="model", title=title, markers=True)  # type: ignore[reportUnknownMemberType]

        fig.update_layout(  # type: ignore[reportUnknownMemberType]
            xaxis_title="Date",
            yaxis_title=metric_name,
            legend_title="Models",
        )

        return fig


__all__ = ["WindowedMetricPlotter"]
