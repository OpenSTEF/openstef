# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Plotting utilities for grouped target metrics visualization.

This module provides plotting capabilities for comparing metrics across
multiple targets and models with optional grouping support.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class GroupedTargetMetricPlotter:
    """Creates bar charts and box plots comparing model metrics across multiple targets.

    This plotter visualizes how different models perform across various forecasting
    targets, making it easy to identify which models work best for specific targets
    or target groups. The resulting charts help answer questions like:

    - Which model has the lowest error for residential vs commercial targets?
    - How consistent is model performance across similar target types?
    - Are there target-specific patterns in model accuracy?

    The plotter can create either grouped bar charts (when no target groups are defined)
    or box plots (when targets are grouped by type/category), providing flexibility
    for different analysis needs.

    Example:
        Basic usage comparing RMSE across targets:

        >>> plotter = GroupedTargetMetricPlotter()
        >>> _ = plotter.add_model("XGBoost",
        ...                       targets=["target_A", "target_B", "target_C"],
        ...                       metric_values=[0.12, 0.15, 0.18])
        >>> _ = plotter.add_model("Random Forest",
        ...                       targets=["target_A", "target_B", "target_C"],
        ...                       metric_values=[0.14, 0.13, 0.20])
        >>> fig = plotter.plot(title="RMSE by Target", metric_name="RMSE")
        >>> type(fig).__name__
        'Figure'

        With target grouping:

        >>> plotter2 = GroupedTargetMetricPlotter()
        >>> _ = plotter2.add_model("XGBoost",
        ...                        targets=["target_A", "target_B"],
        ...                        metric_values=[0.12, 0.15])
        >>> _ = plotter2.set_target_groups({"target_A": "Residential",
        ...                                 "target_B": "Commercial"})
        >>> fig2 = plotter2.plot(title="RMSE by Target Group", metric_name="RMSE")
        >>> type(fig2).__name__
        'Figure'
    """

    def __init__(self):
        """Initialize the plotter with empty data containers."""
        # Models data contains the model name, targets, and metric values
        self.models_data: list[dict[str, str | list[str] | list[float]]] = []
        # Optional mapping from target to target group
        self.target_groups: dict[str, str] | None = None

    def add_model(
        self,
        model_name: str,
        targets: list[str],
        metric_values: list[float],
    ) -> "GroupedTargetMetricPlotter":
        """Add a model's metric values for different targets to the plot.

        Args:
            model_name (str): The name of the model.
            targets (list[str]): List of target names for the x-axis.
            metric_values (list[float]): List of metric values for the y-axis.

        Returns:
            GroupedTargetMetricPlotter: The current instance for method chaining.

        Raises:
            ValueError: If targets and metric_values have different lengths.
        """
        if len(targets) != len(metric_values):
            msg = "Targets and metric values must have the same length"
            raise ValueError(msg)

        model_data = {
            "model": model_name,
            "target": targets,
            "metric_value": metric_values,
        }

        self.models_data.append(model_data)
        return self

    def set_target_groups(self, target_to_group_map: dict[str, str]) -> "GroupedTargetMetricPlotter":
        """Set the mapping from targets to target groups.

        Args:
            target_to_group_map (dict[str, str]): Dictionary mapping target names to their group names.

        Returns:
            GroupedTargetMetricPlotter: The current instance for method chaining.
        """
        self.target_groups = target_to_group_map
        return self

    def plot(self, title: str = "Metric by Target", metric_name: str = "Metric") -> go.Figure:
        """Create and return a plot of metrics across targets for all added models.

        If target_groups is set, creates boxplots grouped by target groups.
        Otherwise creates a grouped bar chart with individual targets.

        Args:
            title (str): Main title of the plot.
            metric_name (str): Name of the metric being plotted.

        Returns:
            plotly.graph_objects.Figure: The resulting plot.

        Raises:
            ValueError: If no model data has been added or if models contain
                targets that are not in the target groups mapping.
        """
        if not self.models_data:
            msg = "No model data has been added. Use add_model first."
            raise ValueError(msg)

        # Combine all model data into a single DataFrame
        model_df_list: list[pd.DataFrame] = []
        for model_data in self.models_data:
            model_df = pd.DataFrame({
                "model": [model_data["model"]] * len(model_data["target"]),
                "target": model_data["target"],
                "metric_value": model_data["metric_value"],
            })
            model_df_list.append(model_df)

        models_df = pd.concat(model_df_list, ignore_index=True)

        # Add target groups if provided
        if self.target_groups:
            models_df["target_group"] = models_df["target"].map(self.target_groups)
            # Check if any targets weren't in the mapping
            missing_targets = models_df[models_df["target_group"].isna()]["target"].unique()
            if len(missing_targets) > 0:
                msg = f"Some targets are missing from the target group mapping: {missing_targets}"
                raise ValueError(msg)

        if self.target_groups:
            # Create boxplot grouped by target groups
            fig = px.box(  # type: ignore[reportUnknownMemberType]
                models_df,
                x="target_group",
                y="metric_value",
                color="model",
                title=title,
            )
        else:
            # Create grouped bar chart with individual targets
            fig = px.bar(models_df, x="target", y="metric_value", color="model", title=title, barmode="group")  # type: ignore[reportUnknownMemberType]

        fig.update_layout(  # type: ignore[reportUnknownMemberType]
            xaxis_title="Target group" if self.target_groups else "Target",
            yaxis_title=metric_name,
            legend_title="Models",
        )

        return fig


__all__ = ["GroupedTargetMetricPlotter"]
