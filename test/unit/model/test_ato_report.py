# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta
from test.utils import BaseTestCase, TestData
from unittest.mock import MagicMock, patch

import pandas as pd
import pytz

from openstf.model.ato_report import make_report_pj

load = pd.DataFrame(
    data={"load": 5}, index=[datetime.utcnow().replace(tzinfo=pytz.utc)]
)
predicted_load = pd.DataFrame(
    data={"forecast_24.0h": 15},
    index=[datetime.utcnow().replace(tzinfo=pytz.utc) + timedelta(days=1)],
)


def get_database_mock():
    db = MagicMock()
    db.get_load_pid = MagicMock(return_value=load)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load)
    return db


@patch("openstf.model.ato_report.DataBase", get_database_mock)
@patch("openstf.model.ato_report.Path", MagicMock())
@patch("openstf.model.ato_report.ConfigManager", MagicMock())
class TestPredict(BaseTestCase):
    def test_make_report_pj(self):
        pj = TestData.get_prediction_job(pid=60)
        result = make_report_pj(pj)
        self.assertEqual(
            result.columns.to_list(), ["realised_load", "day_ahead_forecast"]
        )


if __name__ == "__main__":
    unittest.main()
