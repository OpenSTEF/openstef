# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Output models for analysis visualizations.

This module defines the output structures for generated visualizations,
including individual visualization outputs and aggregated analysis results.
"""

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from openstef_beam.analysis.models.visualization_aggregation import AnalysisScope
from openstef_beam.evaluation.models import Filtering
from openstef_core.base_model import BaseModel


class VisualizationOutput:
    """A generated visualization from evaluation data.

    Represents a single chart, plot, or visual analysis that can be saved as HTML.
    Contains either a Plotly figure object for interactive visualizations or raw
    HTML content for static displays.

    Example:
        Creating a visualization from a Plotly figure:

        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        >>> viz = VisualizationOutput("my_chart", figure=fig)
        >>> viz.get_file_name()
        'my_chart.html'
    """

    def __init__(self, name: str, figure: go.Figure | None = None, html: str | None = None):
        """Initialize visualization output with either a figure or HTML content.

        Args:
            name: Name identifier for the visualization.
            figure: Plotly figure object, if available.
            html: Raw HTML string content, if available.

        Raises:
            ValueError: If neither figure nor html is provided.
        """
        if figure is None and html is None:
            raise ValueError("Either a plotly figure or an HTML string must be provided.")
        self.name = name
        self.figure = figure
        self.html = html

    def write_html(self, file_path: Path, **kwargs: dict[str, Any]) -> None:
        """Write the plotly figure or HTML string to an HTML file.

        Raises:
            ValueError: If neither figure nor HTML content is available.
        """
        if self.figure is not None:
            self.figure.write_html(file_path, include_plotlyjs="cdn", **kwargs)  # type: ignore[reportUnknownMemberType]
        elif self.html is not None:
            file_path.write_text(self.html, encoding="utf-8")
        else:
            raise ValueError("No figure or HTML to write.")

    def get_file_name(self) -> str:
        """Return the file name for the visualization."""
        return f"{self.name}.html"


class AnalysisOutput(BaseModel):
    """Container for analysis results from the benchmarking pipeline.

    Holds all visualizations generated for a specific analysis scope, organized
    by lead time filtering conditions. This allows comparing model performance
    across different forecasting horizons (e.g., 1-hour vs 24-hour ahead predictions).

    The output structure enables systematic organization of results from benchmark
    runs, making it easy to generate reports that compare multiple models across
    various lead times and targets.

    Attributes:
        scope: Analysis context defining what was analyzed (targets, runs, aggregation)
        visualizations: Generated charts and plots grouped by lead time filtering
    """

    scope: AnalysisScope
    visualizations: dict[Filtering, list[VisualizationOutput]]


__all__ = ["AnalysisOutput", "VisualizationOutput"]
