# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from openstef_beam.analysis.models.visualization_aggregation import AnalysisScope
from openstef_beam.evaluation.models import Filtering
from openstef_core.base_model import BaseModel


class VisualizationOutput:
    """A visualization of an EvaluationSubsetReport.
    Can represent either a plotly figure or a raw HTML string.
    """

    def __init__(self, name: str, figure: go.Figure | None = None, html: str | None = None):
        if figure is None and html is None:
            raise ValueError("Either a plotly figure or an HTML string must be provided.")
        self.name = name
        self.figure = figure
        self.html = html

    def write_html(self, file_path: Path, **kwargs: dict[str, Any]) -> None:
        """Write the plotly figure or HTML string to an HTML file."""
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
    scope: AnalysisScope
    visualizations: dict[Filtering, list[VisualizationOutput]]


__all__ = ["AnalysisOutput", "VisualizationOutput"]
