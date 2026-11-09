# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Line chart visualization for learning curves.

Shows the error of a model after each boosting round on the training data and,
when available, on the validation data, so that overfitting is easy to spot.
"""

import pandas as pd
import plotly.graph_objects as go

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.validation import validate_required_columns


class LearningCurvePlotter(BaseConfig):
    """Creates line chart visualizations of learning curves."""

    @staticmethod
    def plot(curves: pd.DataFrame) -> go.Figure:
        """Generate an interactive line chart of the learning curves.

        Draws one line per column. When a ``validation`` column is present, the
        round with the lowest validation error is marked. A training line that
        keeps falling while the validation line stops improving or rises is a
        sign of overfitting.

        Args:
            curves: Learning curves with the boosting round as index and a
                ``train`` column. An optional ``validation`` column is drawn
                as a second line.

        Returns:
            Plotly Figure with the error per boosting round.
        """
        validate_required_columns(curves, required_columns=["train"])

        figure = go.Figure()
        for column in curves.columns:
            figure.add_trace(  # pyright: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=curves.index,
                    y=curves[column],
                    mode="lines",
                    name=str(column),
                    hovertemplate="round %{x}: %{y:.4g}<extra>" + str(column) + "</extra>",
                )
            )

        if "validation" in curves.columns:
            best_round = curves["validation"].idxmin()
            figure.add_trace(  # pyright: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=[best_round],
                    y=[float(curves["validation"].min())],
                    mode="markers",
                    name=f"best validation (round {best_round})",
                    marker={"size": 11, "symbol": "star"},
                    hovertemplate="round %{x}: %{y:.4g}<extra>best validation</extra>",
                )
            )

        figure.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title="Learning curves",
            xaxis_title="Boosting round",
            yaxis_title="Evaluation metric",
            hovermode="x unified",
        )
        return figure
