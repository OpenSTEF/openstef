# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.utils import BaseTestCase, TestData
from unittest.mock import MagicMock, patch
import unittest

import numpy as np
import pandas as pd

from openstf.tasks.calculate_kpi import calc_kpi_for_specific_pid

# Get test data
predicted_load = TestData.load("calculate_kpi_predicted_load.csv")
realised_load = TestData.load("calculate_kpi_relealised_load.csv")

# Prepare dataframe with nans to test low completeness
realised_load_nan = realised_load.copy()
# realised_load_nan.loc[:] = np.nan
realised_load_nan.loc[realised_load_nan.sample(frac=0.5).index, :] = np.NaN

# Prepare dataframe with nans to test low completeness
predicted_load_nan = predicted_load.copy()
predicted_load_nan.loc[predicted_load_nan.sample(frac=0.5).index, :] = np.NaN

# Prepare Database mocks


def get_database_mock():
    db = MagicMock()
    db.get_load_pid = MagicMock(return_value=realised_load)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load)
    db.get_prediction_job = MagicMock(return_value={"id": 295})
    return db


def get_database_mock_realised_nan():
    db = MagicMock()
    db.get_load_pid = MagicMock(return_value=realised_load_nan)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load)
    db.get_prediction_job = MagicMock(return_value={"id": 295})
    return db


def get_database_mock_predicted_nan():
    db = MagicMock()
    db.get_load_pid = MagicMock(return_value=realised_load)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load_nan)
    db.get_prediction_job = MagicMock(return_value={"id": 295})
    return db


class TestPerformanceCalcKpiForSpecificPid(BaseTestCase):

    # Test whether correct kpis are calculated for specific test data
    @patch("openstf.tasks.calculate_kpi'.DataBase", get_database_mock)
    def test_calc_kpi_for_specific_pid(self):
        kpis = calc_kpi_for_specific_pid({"id": 295})
        # use this to store new kpis
        # json.dump(kpis, open(filename, "w"), default=str)
        kpis_ref = TestData.load("calculate_kpi_kpi.json")

        # convert to dataframe to make comparison easier
        self.assertDataframeEqual(
            pd.DataFrame(kpis).drop("date"),
            pd.DataFrame(kpis_ref).drop("date"),
            check_like=True,
        )

    # Test whether none is returned in case of poor completeness for realisex data
    @patch("openstf.tasks.calculate_kpi'.DataBase", get_database_mock_realised_nan)
    def test_calc_kpi_for_specific_pid_poor_completeness_realized(self):
        kpis = calc_kpi_for_specific_pid({"id": 295})
        t_ahead_keys = kpis.keys()
        self.assertIs(kpis[list(t_ahead_keys)[0]]["rMAE"], np.NaN)

    # Test whether none is returned in case of poor completeness for realisex data

    @patch("openstf.tasks.calculate_kpi'.DataBase", get_database_mock_predicted_nan)
    def test_calc_kpi_for_specific_pid_poor_completeness_predicted(self):
        kpis = calc_kpi_for_specific_pid({"id": 295})

        t_ahead_keys = kpis.keys()
        self.assertIs(kpis[list(t_ahead_keys)[0]]["rMAE"], np.NaN)


# Run all tests
if __name__ == "__main__":
    unittest.main()
