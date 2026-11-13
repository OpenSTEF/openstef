# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Visualizations for ensemble model-level contributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from openstef_core.datasets.validated_datasets import ENSEMBLE_COLUMN_SEP
from openstef_core.types import Q, Quantile
from openstef_models.explainability.plotters.common import (
    VerticalTwoPanelPlotLayout,
)

if TYPE_CHECKING:
    from openstef_core.datasets import ForecastDataset, TimeSeriesDataset


class ModelContributionsPlotter:
    """Visualize learned per-model weights alongside the ensemble forecast."""

    @staticmethod
    def plot_stacked_area(
        contributions: TimeSeriesDataset,
        forecast: ForecastDataset,
        quantile: Quantile = Q(0.5),
        target_column: str = "load",
    ) -> go.Figure:
        """Create a stacked-area plot of per-model weights with the ensemble prediction.

        The top panel shows the ensemble prediction for the requested quantile.
        The bottom panel shows each base model's selection weight as a stacked
        area (summing to 1.0), so you can see how the combiner distributes
        weight across models at each timestep.

        Args:
            contributions: Per-model selection weights from a learned-weights
                ``EnsembleForecastingModel``. Use the result of
                ``predict_contributions()`` or the second element of
                ``predict_with_contributions()``. Columns contain model weights
                for each forecast quantile.
            forecast: Output of
                ``EnsembleForecastingModel.predict()`` or the first element
                of ``EnsembleForecastingModel.predict_with_contributions()``.
                Must contain a column for the requested quantile.
            quantile: Forecast quantile to show in both panels. Default ``Q(0.5)``.
            target_column: Name of the target column to exclude from
                contributions. Default ``"load"``.

        Returns:
            Plotly Figure with a prediction line (top) and stacked-area model
            weights (bottom).

        Raises:
            ValueError: If no contribution columns match the requested quantile.
            KeyError: If the forecast dataset has no column for the requested
                quantile.
        """
        df = contributions.data.drop(columns=[target_column], errors="ignore")

        available_columns: dict[tuple[str, str], str] = {}
        for col in df.columns:
            model, separator, model_quantile = col.rpartition(ENSEMBLE_COLUMN_SEP)
            if separator:
                available_columns[(model, model_quantile)] = col

        selected_columns = [
            col for (model, model_quantile), col in available_columns.items() if model_quantile == quantile.format()
        ]
        trace_names = [col[: -len(ENSEMBLE_COLUMN_SEP + quantile.format())] for col in selected_columns]

        if not selected_columns:
            available = sorted({q for _, q in available_columns})
            msg = f"No columns found for quantile {quantile.format()!r}. Available quantiles: {available}."
            raise ValueError(msg)

        selected = df[selected_columns]

        forecast_series = forecast.data[quantile.format()]

        fig = VerticalTwoPanelPlotLayout.make_two_panel_time_figure()
        VerticalTwoPanelPlotLayout.add_top_panel_line_plot(fig, forecast_series.index, forecast_series)

        for col, name in zip(selected_columns, trace_names, strict=True):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=selected[col],
                    mode="lines",
                    stackgroup="contrib",
                    name=name,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            yaxis_title=f"Prediction ({quantile.format().removeprefix('quantile_')})",
            yaxis2_title=f"Model weight ({quantile.format().removeprefix('quantile_')})",
            xaxis2_title="Time",
            margin={"t": 30, "r": 10, "b": 40, "l": 60},
        )

        return fig
