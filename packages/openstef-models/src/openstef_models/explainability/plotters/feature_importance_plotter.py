# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Interactive treemap visualization for feature importance scores.

Creates color-coded treemaps showing relative importance of features in
forecasting models.
"""

import pandas as pd
import plotly.graph_objects as go

from openstef_core.base_model import BaseConfig
from openstef_core.datasets.validation import validate_required_columns
from openstef_core.types import Q, Quantile


class FeatureImportancePlotter(BaseConfig):
    """Creates treemap visualizations of feature importance scores."""

    @staticmethod
    def plot(scores: pd.DataFrame, quantile: Quantile = Q(0.5)) -> go.Figure:
        """Generate interactive treemap showing feature importance.

        Creates a color-coded treemap where each box size and color intensity
        represents the relative importance of a feature. Useful for quickly
        identifying which features contribute most to model predictions.

        Args:
            scores: Feature importance scores with feature names as index and
                quantiles as columns (e.g., 'q0.5', 'q0.95'). Values should be
                normalized to sum to 1.0.
            quantile: Which quantile column to visualize. Defaults to median (0.5).

        Returns:
            Plotly Figure containing interactive treemap with hover information.
            Larger boxes and darker green colors indicate higher importance.
        """
        quantile_column = quantile.format()
        validate_required_columns(scores, required_columns=[quantile_column])

        return go.Figure(
            go.Treemap(
                labels=scores.index,
                parents=pd.Series(data=["Feature importance"] * len(scores), index=scores.index),
                values=scores[quantile_column],
                marker={"colors": scores[quantile_column], "colorscale": "greens"},
                hovertemplate=("<b>%{label}</b><br>importance: %{value:.1%}<extra></extra>"),
            ),
            layout={
                "margin": {
                    "t": 0,
                    "r": 0,
                    "b": 0,
                    "l": 0,
                }
            },
        )
