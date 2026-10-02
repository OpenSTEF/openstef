# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    import pandas as pd


def make_two_panel_time_figure() -> go.Figure:
    return make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8], vertical_spacing=0.03)


def add_top_line(
    fig: go.Figure, x: pd.Index, y: pd.Series, *, name: str = "Prediction", show_legend: bool = False
) -> None:
    return fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line={"color": "black", "width": 1.5}, name=name, showlegend=show_legend),
        row=1,
        col=1,
    )
