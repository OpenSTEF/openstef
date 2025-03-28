# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from openstef.plotting.load_forecast_plotter import LoadForecastPlotter


class TestQuantilePlot(unittest.TestCase):
    def test_quantile_plot_creates_figure_with_expected_traces(self):
        # Arrange
        # Create mock data
        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(24)]

        realized = pd.Series(np.random.normal(100, 10, 24), index=dates)
        forecast = pd.Series(np.random.normal(100, 5, 24), index=dates)

        quantiles_data = {
            "quantile_P10": np.random.normal(80, 5, 24),
            "quantile_P25": np.random.normal(90, 5, 24),
            "quantile_P50": np.random.normal(100, 5, 24),
            "quantile_P75": np.random.normal(110, 5, 24),
            "quantile_P90": np.random.normal(120, 5, 24),
        }
        quantiles = pd.DataFrame(quantiles_data, index=dates)

        # Act
        plot = LoadForecastPlotter()
        figure = plot.plot(realized, forecast, quantiles)

        # Assert
        self.assertIsNotNone(figure)

        trace_names = [trace.name for trace in figure.data]
        self.assertIn("10%-90%", trace_names)
        self.assertIn("25%-75%", trace_names)
        self.assertIn("Forecast (50th)", trace_names)
        self.assertIn("Realized", trace_names)

        self.assertEqual(len(figure.data), 6)
