# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Visualizations for per-sample feature contributions (SHAP values)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots  # pyright: ignore[reportUnknownVariableType]

from openstef_core.datasets import TimeSeriesDataset  # noqa: TC001  # runtime needed for pyright

if TYPE_CHECKING:
    import pandas as pd


class ContributionsPlotter:
    """Visualizations for per-timestep feature contributions."""

    @staticmethod
    def plot_heatmap(
        contributions: TimeSeriesDataset,
        top_n: int = 10,
        target_column: str = "load",
        bias_column: str = "bias",
        *,
        show_prediction: bool = True,
    ) -> go.Figure:
        """Create an interactive heatmap of feature contributions over time.

        X-axis is the prediction datetime, Y-axis shows feature names ranked by mean absolute contribution
        (most important at top). Color ranges from blue (negative) through white (zero) to red (positive).
        When ``show_prediction`` is True a line plot of the model prediction (sum of contributions + bias)
        is shown above the heatmap.

        Args:
            contributions: Output of ``predict_contributions()``.
            top_n: Number of top features to show (ranked by mean absolute contribution).
            target_column: Name of the target column to exclude. Default "load".
            bias_column: Name of the bias column. Default "bias".
            show_prediction: If True, add a prediction line subplot above the heatmap. Default True.

        Returns:
            Plotly Figure with a diverging heatmap centered at zero (and optional prediction line).
        """
        bias = contributions.data[bias_column] if bias_column in contributions.data.columns else None
        cols_to_drop = [c for c in [target_column, bias_column] if c in contributions.data.columns]
        df = contributions.data.drop(columns=cols_to_drop)
        ranked: list[str] = df.abs().mean().sort_values(ascending=False).head(top_n).index.tolist()

        # Most-important feature at top of Y-axis
        y_labels = list(reversed(ranked))

        heatmap = go.Heatmap(
            z=df[y_labels].T.values,
            x=df.index,
            y=y_labels,
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "Contribution"},
            showlegend=False,
        )

        if show_prediction:
            prediction = df.sum(axis=1)
            if bias is not None:
                prediction += bias

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.2, 0.8],
                vertical_spacing=0.03,
            )

            fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=df.index,
                    y=prediction,
                    mode="lines",
                    name="Prediction",
                    line={"color": "black", "width": 1.5},
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(heatmap, row=2, col=1)  # pyright: ignore[reportUnknownMemberType]

            fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
                yaxis_title="Prediction",
                yaxis2_title="Feature",
                xaxis2_title="Time",
                margin={"t": 30, "r": 10, "b": 40, "l": 120},
            )
        else:
            fig = go.Figure(
                data=heatmap,
                layout={
                    "xaxis_title": "Time",
                    "yaxis_title": "Feature",
                    "margin": {"t": 30, "r": 10, "b": 40, "l": 120},
                },
            )

        return fig

    @staticmethod
    def plot_waterfall(
        contributions: TimeSeriesDataset,
        timestep: int = 0,
        top_n: int = 10,
        target_column: str = "load",
        bias_column: str = "bias",
    ) -> go.Figure:
        """Create a waterfall chart decomposing a single timestep's prediction.

        Shows how the bias (base value) is pushed up or down by each feature's
        contribution to arrive at the final prediction.

        Args:
            contributions: Output of ``predict_contributions()``.
            timestep: Row index (0-based) of the timestep to explain.
            top_n: Number of top features to show. Remaining features are
                aggregated into an "other" bar.
            target_column: Name of the target column to exclude. Default "load".
            bias_column: Name of the bias column used as base value. Default "bias".

        Returns:
            Plotly Figure with waterfall chart.
        """
        bias = contributions.data[bias_column] if bias_column in contributions.data.columns else None
        cols_to_drop = [c for c in [target_column, bias_column] if c in contributions.data.columns]
        df = contributions.data.drop(columns=cols_to_drop)
        row = df.iloc[timestep]
        base_value = float(bias.iloc[timestep]) if bias is not None else 0.0

        # Rank by |contribution| for this specific timestep
        abs_sorted = row.abs().sort_values(ascending=False)
        top = abs_sorted.head(top_n).index.tolist()
        remaining = [c for c in abs_sorted.index if c not in top]

        names: list[str] = [bias_column]
        values: list[float] = [base_value]
        measures: list[str] = ["absolute"]

        for feat in top:
            names.append(feat)
            values.append(float(row[feat]))  # pyright: ignore[reportArgumentType]
            measures.append("relative")

        if len(remaining) > 0:
            other_sum = float(row[remaining].sum())
            names.append(f"other ({len(remaining)})")
            values.append(other_sum)
            measures.append("relative")

        names.append("Prediction")
        values.append(base_value + float(row.sum()))
        measures.append("total")

        timestamp = contributions.data.index[timestep]
        return go.Figure(
            go.Waterfall(
                x=names,
                y=values,
                measure=measures,
                connector={"line": {"color": "grey", "width": 0.5}},
                increasing={"marker": {"color": "#ff4136"}},
                decreasing={"marker": {"color": "#0074d9"}},
                totals={"marker": {"color": "#2ecc40"}},
                textposition="outside",
                text=[f"{v:+.4f}" if m == "relative" else f"{v:.4f}" for v, m in zip(values, measures, strict=True)],
            ),
            layout={
                "title": f"Contributions at {timestamp}",
                "yaxis_title": "Contribution",
                "margin": {"t": 50, "r": 10, "b": 40, "l": 60},
                "showlegend": False,
            },
        )

    @staticmethod
    def plot_bar(
        contributions: TimeSeriesDataset,
        top_n: int = 10,
        target_column: str = "load",
        bias_column: str = "bias",
    ) -> go.Figure:
        """Create a horizontal bar chart of mean absolute contributions per feature.

        Features are ranked from most to least important (top to bottom).

        Args:
            contributions: Output of ``predict_contributions()``.
            top_n: Number of top features to show.
            target_column: Name of the target column to exclude. Default "load".
            bias_column: Name of the bias column to exclude. Default "bias".

        Returns:
            Plotly Figure with horizontal bar chart.
        """
        cols_to_drop = [c for c in [target_column, bias_column] if c in contributions.data.columns]
        df = contributions.data.drop(columns=cols_to_drop)
        mean_abs: pd.Series = df.abs().mean().sort_values(ascending=False).head(top_n)

        # Reverse for plotly (bottom-to-top rendering)
        mean_abs = mean_abs.iloc[::-1]

        return go.Figure(
            go.Bar(
                x=mean_abs.values,  # pyright: ignore[reportArgumentType]
                y=mean_abs.index.tolist(),
                orientation="h",
                marker_color="#1f77b4",
                hovertemplate="<b>%{y}</b><br>mean |SHAP|: %{x:.4f}<extra></extra>",
            ),
            layout={
                "xaxis_title": "mean |SHAP value|",
                "yaxis_title": "Feature",
                "margin": {"t": 30, "r": 10, "b": 40, "l": 120},
                "showlegend": False,
            },
        )
