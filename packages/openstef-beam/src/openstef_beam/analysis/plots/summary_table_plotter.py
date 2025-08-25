# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd


class SummaryTablePlotter:
    """Class to plot summary tables."""

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
        styled_html = (
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

        return styled_html


__all__ = ["SummaryTablePlotter"]
