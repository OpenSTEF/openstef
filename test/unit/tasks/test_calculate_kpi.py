# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from openstef.exceptions import NoPredictedLoadError, NoRealisedLoadError
from openstef.tasks.calculate_kpi import calc_kpi_for_specific_pid

# Get test data
predicted_load = TestData.load("calculate_kpi_predicted_load.csv")
realised_load = TestData.load("calculate_kpi_relealised_load.csv")

# Prepare dataframe with nans to test low completeness
realised_load_nan = realised_load.copy()
realised_load_nan.loc[realised_load_nan.sample(frac=0.5).index, :] = np.nan

# Prepare dataframe with nans to test low completeness
predicted_load_nan = predicted_load.copy()
predicted_load_nan.loc[predicted_load_nan.sample(frac=0.5).index, :] = np.nan

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
        kpis = calc_kpi_for_specific_pid(
            prediction_job["id"], realised_load, predicted_load, realised_load
        )
        # Remove date
        # type(date)==datetime gave a LOT of errors
        kpis = {
            t_ahead: {kpi: value for kpi, value in kpi_dict.items() if kpi != "date"}
            for t_ahead, kpi_dict in kpis.items()
        }

        # Use line below to store new kpis
        # TestData.save(kpis, 'calculate_kpi_kpi.json')
        kpis_ref = TestData.load("calculate_kpi_kpi.json")

        # convert to dataframe to make comparison easier
        self.assertDataframeEqual(
            pd.DataFrame(kpis).reindex(sorted(pd.DataFrame(kpis).columns), axis=1),
            pd.DataFrame(kpis_ref).reindex(
                sorted(pd.DataFrame(kpis_ref).columns), axis=1
            ),
            check_like=True,
        )

    # Test whether none is returned in case of poor completeness for realised data
    def test_calc_kpi_for_specific_pid_poor_completeness_realized(self):
        kpis = calc_kpi_for_specific_pid(
            prediction_job["id"], realised_load_nan, predicted_load, realised_load_nan
        )
        t_ahead_keys = kpis.keys()
        self.assertIs(kpis[list(t_ahead_keys)[0]]["rMAE"], np.nan)

    # Test whether none is returned in case of poor completeness for predicted data
    def test_calc_kpi_for_specific_pid_poor_completeness_predicted(self):
        kpis = calc_kpi_for_specific_pid(
            prediction_job["id"], realised_load, predicted_load_nan, realised_load
        )

        t_ahead_keys = kpis.keys()
        self.assertIsNAN(kpis[list(t_ahead_keys)[0]]["NSME"])

    def test_calc_kpi_for_specific_pid_constant_load(self):
        """If load is constant, a warning should be raised, but kpi's should still be calculated"""
        realised_load_constant = realised_load.copy()
        realised_load_constant.iloc[1:, :] = realised_load_constant.iloc[0, :]
        kpis = calc_kpi_for_specific_pid(
            prediction_job["id"], realised_load_constant, predicted_load, realised_load
        )
        self.assertIsNAN(kpis["4.0h"]["MAE"])  # arbitrary time horizon tested
        self.assertAlmostEqual(kpis["4.0h"]["MAE"], 2.9145, places=3)

    def test_calc_kpi_no_load_exception(self):
        """Assert that correct exceptions are raised for empty load."""

        with self.assertRaises(NoRealisedLoadError):
            calc_kpi_for_specific_pid(
                prediction_job["id"], pd.DataFrame(), predicted_load, pd.DataFrame()
            )

    def test_calc_kpi_no_prediction_exception(self):
        """Assert that correct exceptions are raised for empty prediction."""

        with self.assertRaises(NoPredictedLoadError):
            calc_kpi_for_specific_pid(
                prediction_job["id"], realised_load, pd.DataFrame(), realised_load
            )

    def test_calc_kpi_empty_basecase(self):
        """An empty basecase should not return an exception
        Instead, basecase metrics should be NaN"""

        kpis = calc_kpi_for_specific_pid(
            prediction_job["id"], realised_load, predicted_load, basecase=pd.DataFrame()
        )

        # Assert KPIs
        assert isinstance(kpis, dict)
        arbitrary_tAhead = "47.0h"
        self.assertEqual(kpis[arbitrary_tAhead]["skill_score_basecase"], 0)
