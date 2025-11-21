# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Probability calibration plotting for forecast uncertainty validation.

This module provides specialized plotting for validating probabilistic forecasts
by comparing predicted probabilities with observed frequencies.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from openstef_core.types import Quantile


class QuantileProbabilityPlotter:
    """Creates calibration plots comparing forecasted vs observed probabilities.

    This plotter validates probabilistic forecasts by examining whether predicted
    uncertainty levels match actual outcomes. The resulting plots help answer:

    - Are 90% confidence intervals correct 90% of the time?
    - Do models over-estimate or under-estimate forecast uncertainty?
    - Which model provides the most reliable uncertainty quantification?
    - How well-calibrated are different quantile predictions?

    The plots show:
    - Scatter plots of forecasted vs observed probabilities
    - Perfect calibration diagonal line for reference
    - Model-specific calibration curves
    - Visual indicators of over/under-confidence

    Well-calibrated models will have points close to the diagonal line, while
    systematic deviations indicate bias in uncertainty estimation.

    Example:
        Validating forecast calibration:

        >>> from openstef_core.types import Quantile
        >>> plotter = QuantileProbabilityPlotter()
        >>>
        >>> # Add model calibration data
        >>> forecasted = [Quantile(0.1), Quantile(0.3), Quantile(0.5), Quantile(0.9)]
        >>> observed = [Quantile(0.12), Quantile(0.28), Quantile(0.52), Quantile(0.88)]
        >>> _ = plotter.add_model("XGBoost", forecasted, observed)
        >>> # Generate calibration plot
        >>> fig = plotter.plot(title="Forecast Calibration Analysis")
        >>> type(fig).__name__
        'Figure'
    """

    def __init__(self):
        """Initialize the plotter with empty model data storage."""
        # Model data contains the model name, forecasted probabilities, and observed probabilities
        self.models_data: list[dict[str, str | list[Quantile]]] = []

    def add_model(
        self,
        model_name: str,
        forecasted_probs: list[Quantile],
        observed_probs: list[Quantile],
    ) -> "QuantileProbabilityPlotter":
        """Add a model's forecasted and observed probabilities to the plot.

        Args:
            model_name: The name of the model.
            forecasted_probs: List of forecasted probabilities.
            observed_probs: List of observed probabilities.

        Returns:
            QuantileProbabilityPlotter: The current instance for method chaining.

        Raises:
            ValueError: If forecasted and observed probability lists have different lengths.
        """
        if len(forecasted_probs) != len(observed_probs):
            msg = "Forecasted probabilities and observed probabilities must have the same length"
            raise ValueError(msg)

        model_data = {
            "model": model_name,
            "forecasted_prob": forecasted_probs,
            "observed_prob": observed_probs,
        }

        self.models_data.append(model_data)
        return self

    def plot(self, title: str = "Quantile probability plot") -> go.Figure:
        """Create and return a quantile probability plot for all added models.

        Args:
            title (str): Title of the plot.

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
                "model": [model_data["model"]] * len(model_data["forecasted_prob"]),
                "forecasted_prob": model_data["forecasted_prob"],
                "observed_prob": model_data["observed_prob"],
            })
            model_df_list.append(model_df)

        models_df = pd.concat(model_df_list, ignore_index=True)

        # Create the calibration plot
        fig = px.line(  # type: ignore[reportUnknownMemberType]
            models_df,
            x="forecasted_prob",
            y="observed_prob",
            color="model",
            markers=True,
            title=title,
        )

        # Add the perfect calibration line (y=x)
        fig.add_trace(  # type: ignore[reportUnknownMemberType]
            go.Scattergl(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                name="Perfect probability",
                showlegend=True,
            )
        )

        fig.update_layout(  # type: ignore[reportUnknownMemberType]
            xaxis_title="Forecasted Probability",
            yaxis_title="Observed Probability",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1.05]},
            legend_title="Models",
        )

        return fig


__all__ = ["QuantileProbabilityPlotter"]
