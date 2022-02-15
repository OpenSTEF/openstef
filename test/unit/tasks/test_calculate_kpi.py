# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from openstef.exceptions import NoPredictedLoadError, NoRealisedLoadError
from openstef.tasks.calculate_kpi import calc_kpi_for_specific_pj

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

prediction_job = TestData.get_prediction_job(pid=307)


# Prepare Database mocks
def get_database_mock():
    db = MagicMock()
    db.get_load_pid = MagicMock(return_value=realised_load)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load)
    db.get_prediction_job = MagicMock(return_value=prediction_job)
    return db


def get_database_mock_realised_nan():
    db = MagicMock()
    db.get_load_pid = MagicMock(return_value=realised_load_nan)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load)
    db.get_prediction_job = MagicMock(return_value=prediction_job)
    return db


def get_database_mock_predicted_nan():
    db = MagicMock()
    db.get_load_pid = MagicMock(return_value=realised_load)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load_nan)
    db.get_prediction_job = MagicMock(return_value=prediction_job)
    return db


def get_database_mock_realised_constant():
    db = MagicMock()
    realised_load_constant = realised_load.copy()
    realised_load_constant.iloc[1:, :] = realised_load_constant.iloc[0, :]
    db.get_load_pid = MagicMock(return_value=realised_load_constant)
    db.get_predicted_load_tahead = MagicMock(return_value=predicted_load)
    db.get_prediction_job = MagicMock(return_value=prediction_job)
    return db


def get_database_mock_realised_empty():
    db = get_database_mock_realised_constant()
    db.get_load_pid.return_value = pd.DataFrame()
    return db


def get_database_mock_predicted_empty():
    db = get_database_mock_predicted_nan()
    db.get_predicted_load_tahead.return_value = pd.DataFrame()
    return db


class TestPerformanceCalcKpiForSpecificPid(BaseTestCase):

    # Test whether correct kpis are calculated for specific test data
    def test_calc_kpi_for_specific_pid(self):
        kpis = calc_kpi_for_specific_pj(
            prediction_job, realised_load, predicted_load, realised_load
        )
        # use this to store new kpis
        # json.dump(kpis, open(filename, "w"), default=str)
        kpis_ref = TestData.load("calculate_kpi_kpi.json")

        # convert to dataframe to make comparison easier
        self.assertDataframeEqual(
            pd.DataFrame(kpis)
            .drop("date")
            .reindex(sorted(pd.DataFrame(kpis).drop("date").columns), axis=1),
            pd.DataFrame(kpis_ref)
            .drop("date")
            .reindex(sorted(pd.DataFrame(kpis_ref).drop("date").columns), axis=1),
            check_like=True,
        )

    # Test whether none is returned in case of poor completeness for realised data
    def test_calc_kpi_for_specific_pid_poor_completeness_realized(self):
        kpis = calc_kpi_for_specific_pj(
            prediction_job, realised_load_nan, predicted_load, realised_load_nan
        )
        t_ahead_keys = kpis.keys()
        self.assertIs(kpis[list(t_ahead_keys)[0]]["rMAE"], np.NaN)

    # Test whether none is returned in case of poor completeness for predicted data
    def test_calc_kpi_for_specific_pid_poor_completeness_predicted(self):
        kpis = calc_kpi_for_specific_pj(
            prediction_job, realised_load, predicted_load_nan, realised_load
        )

        t_ahead_keys = kpis.keys()
        self.assertIs(kpis[list(t_ahead_keys)[0]]["rMAE"], np.NaN)

    def test_calc_kpi_for_specific_pid_constant_load(self):
        """If load is constant, a warning should be raised, but kpi's should still be calculated"""
        realised_load_constant = realised_load.copy()
        realised_load_constant.iloc[1:, :] = realised_load_constant.iloc[0, :]
        kpis = calc_kpi_for_specific_pj(
            prediction_job, realised_load_constant, predicted_load, realised_load
        )
        self.assertIsNAN(kpis["4.0h"]["MAE"])  # arbitrary time horizon tested
        self.assertAlmostEqual(kpis["4.0h"]["MAE"], 2.9145, places=3)

    def test_calc_kpi_no_load_exception(self):
        """Assert that correct exceptions are raised for
        empty load"""

        with self.assertRaises(NoRealisedLoadError):
            calc_kpi_for_specific_pj(
                prediction_job, pd.DataFrame(), predicted_load, pd.DataFrame()
            )

    def test_calc_kpi_no_prediction_exception(self):
        """Assert that correct exceptions are raised for
        empty prediction"""

        with self.assertRaises(NoPredictedLoadError):
            calc_kpi_for_specific_pj(
                prediction_job, realised_load, pd.DataFrame(), realised_load
            )


# Run all tests
if __name__ == "__main__":
    unittest.main()
