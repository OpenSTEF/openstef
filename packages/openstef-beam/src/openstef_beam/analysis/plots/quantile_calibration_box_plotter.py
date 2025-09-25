# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Quantile calibration box plotting for forecast uncertainty validation across multiple targets.

This module provides specialized boxplot visualization for evaluating probabilistic forecast
calibration across multiple targets, showing calibration error distributions per quantile.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from openstef_core.types import Quantile


class QuantileCalibrationBoxPlotter:
    """Creates boxplots showing calibration error distributions across targets for each quantile.

    This plotter visualizes how well probabilistic forecasts are calibrated by showing
    the distribution of calibration errors (observed - forecasted probability) across
    multiple targets for each quantile level. This helps identify:

    - Which quantiles are systematically over/under-confident across targets
    - The consistency of calibration across different targets
    - Overall uncertainty estimation quality per quantile level

    Well-calibrated forecasts will have boxplots centered around zero (no systematic bias)
    with tight distributions (consistent calibration across targets).

    Example:
        Validating forecast calibration across multiple models and targets:

        >>> from openstef_core.types import Quantile
        >>> plotter = QuantileCalibrationBoxPlotter()
        >>>
        >>> # Define common quantiles for all models/targets
        >>> quantiles = [Quantile(0.1), Quantile(0.5), Quantile(0.9)]
        >>>
        >>> # ModelA: Well-calibrated model across different sites
        >>> _ = plotter.add_model("ModelA", quantiles, [Quantile(0.08), Quantile(0.52), Quantile(0.88)])  # Site1
        >>> _ = plotter.add_model("ModelA", quantiles, [Quantile(0.12), Quantile(0.48), Quantile(0.92)])  # Site2
        >>> _ = plotter.add_model("ModelA", quantiles, [Quantile(0.09), Quantile(0.51), Quantile(0.89)])  # Site3
        >>>
        >>> # ModelB: Overconfident model across different sites
        >>> _ = plotter.add_model("ModelB", quantiles, [Quantile(0.15), Quantile(0.55), Quantile(0.85)])  # Site1
        >>> _ = plotter.add_model("ModelB", quantiles, [Quantile(0.18), Quantile(0.58), Quantile(0.82)])  # Site2
        >>> _ = plotter.add_model("ModelB", quantiles, [Quantile(0.16), Quantile(0.57), Quantile(0.84)])  # Site3
        >>>
        >>> # Generate boxplot showing calibration error distributions
        >>> fig = plotter.plot(title="Multi-Model Calibration Analysis")
        >>> type(fig).__name__
        'Figure'
    """

    def __init__(self) -> None:
        """Initialize the plotter with empty model data storage."""
        # Model data contains the model name, forecasted probabilities, and observed probabilities
        self.models_data: list[dict[str, str | list[Quantile]]] = []

    def add_model(
        self,
        model_name: str,
        forecasted_probs: list[Quantile],
        observed_probs: list[Quantile],
    ) -> "QuantileCalibrationBoxPlotter":
        """Add a model's forecasted and observed probabilities to the plot.

        Args:
            model_name: The name of the model.
            forecasted_probs: List of forecasted probabilities.
            observed_probs: List of observed probabilities.

        Returns:
            QuantileCalibrationBoxPlotter: The current instance for method chaining.

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

    def plot(self, title: str = "Quantile Calibration Boxplot") -> go.Figure:
        """Create and return a quantile calibration boxplot for all added models.

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

        # Combine all model data into a single DataFrame with calibration errors
        model_df_list: list[pd.DataFrame] = []
        for model_data in self.models_data:
            # Calculate calibration errors (observed - forecasted)
            calibration_errors = [
                float(obs) - float(fore)
                for obs, fore in zip(model_data["observed_prob"], model_data["forecasted_prob"], strict=True)
            ]

            # Create quantile labels
            quantile_labels = [q.format() for q in model_data["forecasted_prob"]]

            model_df = pd.DataFrame({
                "Model": [model_data["model"]] * len(calibration_errors),
                "Quantile": quantile_labels,
                "Calibration_Error": calibration_errors,
            })
            model_df_list.append(model_df)

        models_df = pd.concat(model_df_list, ignore_index=True)

        # Create the boxplot
        fig: go.Figure = px.box(  # type: ignore[reportUnknownMemberType]
            models_df,
            x="Quantile",
            y="Calibration_Error",
            color="Model",
            title=title,
        )

        # Add a horizontal line at y=0 (perfect calibration)
        fig.add_hline(  # type: ignore[reportUnknownMemberType]
            y=0,
            line_dash="dash",
            line_color="black",
            annotation_text="Perfect Calibration",
            annotation_position="bottom right",
        )

        # Add region labels for over/under estimation
        self._add_over_under_estimation_region_labels(fig)

        # Update layout for better readability
        fig.update_layout(  # type: ignore[reportUnknownMemberType]
            xaxis_title="Quantile Level",
            yaxis_title="Calibration Error (Observed - Forecasted)",
            showlegend=True,
            hovermode="x unified",
        )

        return fig

    @staticmethod
    def _add_over_under_estimation_region_labels(fig: go.Figure) -> None:
        """Add region labels to indicate over/under estimation areas on the plot.

        Args:
            fig: The plotly figure to add annotations to.
        """
        # Add overestimation label (positive calibration error region)
        fig.add_annotation(  # type: ignore[reportUnknownMemberType]
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            text="(Overestimation)",
            showarrow=False,
            font={"size": 10, "color": "gray"},
            xanchor="right",
            yanchor="top",
        )

        # Add underestimation label (negative calibration error region)
        fig.add_annotation(  # type: ignore[reportUnknownMemberType]
            x=1,
            y=0,
            xref="paper",
            yref="paper",
            text="(Underestimation)",
            showarrow=False,
            font={"size": 10, "color": "gray"},
            xanchor="right",
            yanchor="bottom",
        )


__all__ = ["QuantileCalibrationBoxPlotter"]
