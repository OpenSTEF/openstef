# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

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
