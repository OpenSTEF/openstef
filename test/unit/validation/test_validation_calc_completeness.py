# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from test.unit.utils.base import BaseTestCase

import numpy as np
import pandas as pd

from openstef.validation.validation import calc_completeness


class CalcCompletenessTest(BaseTestCase):

    datetime_format = "%Y-%m-%dT%H:%M:%S%z"

    def test_complete_dataframe(self):
        df = pd.DataFrame(index=[0, 1], data={"col1": [1, 1]})
        completeness = calc_completeness(df)

        self.assertEqual(completeness, 1.0)

    def test_APX_missing(self):
        df = pd.DataFrame(index=range(2 * 96), data={"APX": [np.nan] * 2 * 96})
        completeness = calc_completeness(df, time_delayed=True)
        self.assertEqual(completeness, 1 / 2)

    def test_incomplete_dataframe(self):
        df = pd.DataFrame(index=[0, 1, 2], data={"col1": [1, np.nan, 3]})
        completeness = calc_completeness(df)

        self.assertEqual(completeness, 2 / 3)

    def test_weighted_dataframe(self):
        df = pd.DataFrame(index=[0, 1], data={"col1": [1, np.nan], "col2": [3, 4]})
        weights = [1, 2]

        completeness = calc_completeness(df, weights)
        self.assertEqual(completeness, (1 * 0.5 + 2 * 1) / 3)

    def test_timedelayed_dataframe(self):
        df = pd.DataFrame(
            index=[0, 1, 3],
            data={"T-15min": [1, np.nan, np.nan], "T-30min": [2, 3, np.nan]},
        )
        completeness = calc_completeness(df, time_delayed=True)
        self.assertEqual(completeness, 1)

    def test_timedelayed_incomplete_dataframe(self):
        df = pd.DataFrame(
            index=[0, 1, 3],
            data={"T-15min": [1, np.nan, np.nan], "T-30min": [2, np.nan, np.nan]},
        )  # first nan is unexpected
        completeness = calc_completeness(df, time_delayed=True)
        self.assertAlmostEqual(completeness, 1 - 1 / 6, places=3)

    def test_timedelayed_advanced_dataframe(self):
        df = pd.DataFrame(
            index=[0, 1, 3],
            data={
                "T-15min": [1, np.nan, np.nan],
                "T-30min": [2, 3, np.nan],
                "col1": [1, np.nan, 2],
            },
        )
        weights = [1, 1, 2]
        completeness = calc_completeness(df, weights, time_delayed=True)
        self.assertEqual(completeness, (1 + 1 + 2 / 3 * 2) / 4)

    def test_homogenise_timeindex_complete(self):
        df_complete = pd.DataFrame(
            {"aggregated": [10, 20, 30]},
            index=pd.to_datetime(
                ["2019-01-01 10:00:00", "2019-01-01 10:05:00", "2019-01-01 10:10:00"]
            ),
        )
        completeness_df_complete = calc_completeness(df_complete)
        self.assertAlmostEqual(completeness_df_complete, 1)

    def test_homogenise_timeindex_incomplete(self):
        df_incomplete = pd.DataFrame(
            {"aggregated": [10, 20, 30, 40]},
            index=pd.to_datetime(
                [
                    "2019-01-01 10:00:00",
                    "2019-01-01 10:05:00",
                    # Note the missing value
                    "2019-01-01 10:15:00",
                    "2019-01-01 10:20:00",
                ]
            ),
        )
        completeness_df_incomplete = calc_completeness(df_incomplete, homogenise=True)
        completeness_df_incomplete_nothomogenised = calc_completeness(
            df_incomplete, homogenise=False
        )
        self.assertAlmostEqual(completeness_df_incomplete, 0.8)
        self.assertAlmostEqual(completeness_df_incomplete_nothomogenised, 1)

    def test_empty_dataframe(self):
        empty_compl_homogenise = calc_completeness(
            pd.DataFrame(data={"load": []}, index=pd.DatetimeIndex([]))
        )
        empty_compl_nohomogenise = calc_completeness(
            pd.DataFrame(data={"load": []}, index=pd.DatetimeIndex([])),
            homogenise=False,
        )
        self.assertAlmostEqual(empty_compl_homogenise, 0.0)
        self.assertAlmostEqual(empty_compl_nohomogenise, 0.0)

    def test_calc_completeness_no_negatives(self):
        """Test added after bug.
        If time delayed is True, T-7d gave a negative weight,
        falsely resulting in a very low completeness"""
        df = pd.DataFrame(
            index=[0, 1, 3],
            data={
                "T-15min": [1, np.nan, np.nan],
                "T-7d": [2, 3, 4],
                "T-24d": [4, 5, 6],
                "col1": [1, np.nan, 2],
            },
        )
        completeness = calc_completeness(df, time_delayed=True)
        self.assertEqual(completeness, 11 / 12.0)
