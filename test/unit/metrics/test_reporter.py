# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import os
import tempfile
import unittest
from unittest.mock import patch

import plotly.graph_objects as go

from openstef.metrics.reporter import Report, Reporter


class TestReport(unittest.TestCase):
    def test_report(self):
        # Arrange & act
        dummy_report = Report(
            feature_importance_figure=None,
            data_series_figures={},
            metrics={},
            signature=None,
        )
        # Assert
        assert isinstance(dummy_report, Report)


class TestReporter(unittest.TestCase):
    def test_reporter_write_to_disk_written(self):
        # Arrange
        figure = go.Figure(
            data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
            layout=go.Layout(
                title=go.layout.Title(text="A Simple Figure to test writing to disk")
            ),
        )
        report = Report(
            feature_importance_figure=figure,
            data_series_figures={"Predictor0.25": figure, "Predictor47.0": figure},
            metrics={},
            signature=None,
        )

        # Act
        with tempfile.TemporaryDirectory() as temp_model_dir:
            Reporter.write_report_to_disk(report, temp_model_dir)
            figure_files = os.listdir(temp_model_dir)
            # Assert
            assert len(figure_files) == 3
            assert "Predictor0.25.html" in figure_files
            assert "Predictor47.0.html" in figure_files
            assert "weight_plot.html" in figure_files

    @patch("plotly.graph_objects.Figure.write_html")
    def test_reporter_write_to_disk_nothing_written(self, write_html_mock):
        # Arrange
        figure = go.Figure(
            data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
            layout=go.Layout(
                title=go.layout.Title(text="A Simple Figure to test writing to disk")
            ),
        )
        report = Report(
            feature_importance_figure=figure,
            data_series_figures={"Predictor0.25": figure, "Predictor47.0": figure},
            metrics={},
            signature=None,
        )

        # Act
        Reporter.write_report_to_disk(report, None)

        # Assert
        assert write_html_mock.call_count == 0

    def test_reporter_write_to_disk_no_write_without_figures(self):
        # Arrange
        report = Report(
            feature_importance_figure=None,
            data_series_figures={"Predictor0.25": None, "Predictor47.0": None},
            metrics={},
            signature=None,
        )

        # Act
        with tempfile.TemporaryDirectory() as temp_model_dir:
            Reporter.write_report_to_disk(report, temp_model_dir)
            figure_files = os.listdir(temp_model_dir)
            # Assert
            assert len(figure_files) == 0
