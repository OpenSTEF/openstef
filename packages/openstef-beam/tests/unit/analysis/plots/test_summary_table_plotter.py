# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd

from openstef_beam.analysis.plots import SummaryTablePlotter


def test_plot_returns_html_table():
    # Arrange
    data = pd.DataFrame({"columnA": [123, 567], "columnB": ["xyz", "abc"]})

    # Act
    plotter = SummaryTablePlotter(data)
    html = plotter.plot()

    # Assert
    assert isinstance(html, str)
    assert "columnA" in html
    assert "columnB" in html
    assert "123" in html
    assert "567" in html
    assert "xyz" in html
    assert "abc" in html
