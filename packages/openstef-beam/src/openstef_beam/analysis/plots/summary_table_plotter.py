# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Summary table generation for evaluation metrics display.

This module provides formatted HTML table generation from evaluation metrics,
creating professional-looking summary reports for model comparison.
"""

import pandas as pd


class SummaryTablePlotter:
    """Creates formatted HTML tables from evaluation metrics data.

    This plotter transforms evaluation metrics into HTML tables suitable for
    reports and dashboards. The tables help answer questions like:

    - Which model has the best overall performance across all metrics?
    - How do metric values compare between different targets or runs?
    - What are the summary statistics for forecast accuracy?

    The generated tables include:
    - Styled headers and borders for professional appearance
    - Left-aligned text for readability
    - Consistent formatting across all metrics
    - HTML output suitable for embedding in reports

    Example:
        Creating a summary table from metrics:

        >>> import pandas as pd
        >>> metrics_data = pd.DataFrame({
        ...     'Model': ['XGBoost', 'Random Forest', 'LSTM'],
        ...     'RMSE': [0.12, 0.15, 0.14],
        ...     'MAE': [0.08, 0.11, 0.10],
        ...     'R²': [0.85, 0.78, 0.82]
        ... })
        >>> plotter = SummaryTablePlotter(metrics_data)
        >>> html_table = plotter.plot()
        >>> # html_table contains styled HTML ready for display
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize the SummaryTablePlotter with data.

        Args:
            data (pd.DataFrame): DataFrame containing the data to be plotted.
        """
        self.data = data

    def plot(self) -> str:
        """Creates a formatted HTML table from the DataFrame.

        Returns:
            str: A formatted HTML table
        """
        # Apply basic styling to the HTML table
        return (
            self.data.style.set_table_attributes('class="dataframe"')
            .set_properties(**{"text-align": "left", "padding": "5px", "border": "1px solid #ddd"})  # type: ignore[arg-type]
            .set_table_styles([
                {
                    "selector": "thead th",
                    "props": [
                        ("font-weight", "bold"),
                        ("text-align", "left"),
                        ("padding", "8px"),
                        ("border-bottom", "2px solid #ddd"),
                    ],
                },
            ])
            .hide(axis="index")
            .to_html()
        )


__all__ = ["SummaryTablePlotter"]
