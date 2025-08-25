# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from openstef_core.types import Quantile


class PrecisionRecallCurvePlotter:
    """Class to plot the precision against the recall for each quantile, for each model."""

    def __init__(self):
        # Model data contains the model name, precision values, recall values, and quantile values
        self.models_data: list[dict[str, str | list[float] | list[Quantile]]] = []

    def add_model(
        self,
        model_name: str,
        precision_values: list[float],
        recall_values: list[float],
        quantiles: list[Quantile],
    ) -> "PrecisionRecallCurvePlotter":
        """Add a model's precision and recall values to the plot.

        Args:
            model_name (str): The name of the model.
            precision_values (list): List of precision values for each quantile.
            recall_values (list): List of recall values for each quantile.
            quantiles (list, optional): List of quantile values. If None, will use indices.

        Returns:
            PrecisionRecallCurvePlotter: The current instance for method chaining.
        """
        if len(precision_values) != len(recall_values):
            msg = "Precision and recall lists must have the same length"
            raise ValueError(msg)

        if len(quantiles) != len(precision_values):
            msg = "Quantiles list must have the same length as precision and recall lists"
            raise ValueError(msg)

        model_data = {
            "model": model_name,
            "precision": precision_values,
            "recall": recall_values,
            "quantile": quantiles,
        }

        self.models_data.append(model_data)
        return self

    def plot(self, title: str = "Precision-Recall Curve") -> go.Figure:
        """Create and return a precision-recall curve plot for all added models.

        Args:
            title (str): Title of the plot.
            width (int): Width of the plot in pixels.
            height (int): Height of the plot in pixels.

        Returns:
            plotly.graph_objects.Figure: The precision-recall curve plot.
        """
        if not self.models_data:
            msg = "No model data has been added. Use add_model first."
            raise ValueError(msg)

        # Combine all model data into a single DataFrame
        model_df_list: list[pd.DataFrame] = []
        for model_data in self.models_data:
            model_df = pd.DataFrame({
                "model": [model_data["model"]] * len(model_data["precision"]),
                "precision": model_data["precision"],
                "recall": model_data["recall"],
                "quantile": model_data["quantile"],
            })
            model_df_list.append(model_df)

        models_df = pd.concat(model_df_list, ignore_index=True)

        fig = px.line(  # type: ignore - needs stubs
            models_df,
            x="recall",
            y="precision",
            color="model",
            markers=True,
            hover_data=["quantile"],
            title=title,
        )

        fig.update_layout(  # type: ignore - needs stubs
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1.05]},
            legend_title="Models",
        )

        return fig


__all__ = ["PrecisionRecallCurvePlotter"]
