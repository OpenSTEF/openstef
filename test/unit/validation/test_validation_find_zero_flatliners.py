# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-

import unittest
from datetime import datetime, timedelta

import pandas as pd

from openstf.validation.validation import find_zero_flatliner
from test.utils import BaseTestCase

now = datetime.utcnow()
date_rng = pd.date_range(start=now - timedelta(minutes=30), end=now, freq="0.25H")
date_rng2 = pd.date_range(start=now - timedelta(minutes=60), end=now, freq="0.25H")
date_rng3 = pd.date_range(start=now - timedelta(minutes=30), end=now, freq="0.25H")

df_no_flatliner = pd.DataFrame(
    {"date": date_rng, "col1": [8.16, 7.64, 7.44], "col2": [8.16, 7.64, 7.44]}
)
df_flatliner = pd.DataFrame(
    {"date": date_rng2, "col1": [8.0, 8.0, 8.0, 8.0, 8.0], "col2": [8.0, 0, 0, 0, 8.0]}
)
df_flatline_at_end = pd.DataFrame(
    {"date": date_rng2, "col1": [8.0, 8.0, 8.0, 8.0, 8.0], "col2": [8.0, 0, 0, 0, 0]}
)
df_flatline_at_start = pd.DataFrame(
    {"date": date_rng2, "col1": [8.0, 8.0, 8.0, 8.0, 8.0], "col2": [0, 0, 0, 8.0, 8.0]}
)
df_compensated_flatliner = pd.DataFrame(
    {"date": date_rng2, "col1": [8.0, 16, 16, 16, 8.0], "col2": [8.0, 0, 0, 0, 8.0]}
)
df_nan_value = pd.DataFrame(
    {
        "date": date_rng2,
        "col1": [8.0, 8.0, 8.0, 8.0, 8.0],
        "col2": [None, None, None, None, None],
    }
)
df_some_nan = pd.DataFrame(
    {"date": date_rng2, "col1": [8.0, 8.0, 8.0, 8.0, 8.0], "col2": [8.0, 0, 0, 0, None]}
)
df_zero_file = pd.DataFrame({"date5": [], "col1": [], "col2": []})
df_all_zerovalues = pd.DataFrame(
    {"date6": date_rng, "col1": [8.16, 7.64, 7.44], "col2": [0, 0, 0]}
)


df_no_flatliner = df_no_flatliner.set_index("date")
df_flatliner = df_flatliner.set_index("date")
df_compensated_flatliner = df_compensated_flatliner.set_index("date")
df_nan_value = df_nan_value.set_index("date")
df_some_nan = df_some_nan.set_index("date")
df_zero_file = df_zero_file.set_index("date5")
df_all_zerovalues = df_all_zerovalues.set_index("date6")


class TestValidationFindZeroFlatliners(BaseTestCase):
    def test_no_zero_values(self):
        """Data: no zero-values in any of the trafo's

        Expected: empty list, since the function only detects zero values (all trafo's)
        """
        df = df_no_flatliner
        threshold = 0.25
        expected = None
        result = find_zero_flatliner(df, threshold)

        self.assertEqual(expected, result)

    def test_all_zerovalues(self):
        """Data: one trafo containing all zero_values

        Expected: empty list, since there is no data to check
        """
        df = df_all_zerovalues
        threshold = 0.25
        result = find_zero_flatliner(df, threshold)
        expected = None
        self.assertEqual(result, expected)

    def test_all_NAN(self):
        """Data: one trafo containing NaN values

        Expected: empty list, since there is no data to check
        """
        df = df_nan_value
        threshold = 0.25
        result = find_zero_flatliner(df, threshold)
        expected = None
        self.assertEqual(result, expected)

    def test_find_zero_flatliner(self):
        """Data: zero-values at one trafo which is not compensated, only at start.

        Expected: None, since
        """
        df = df_flatliner
        threshold = 0.25
        result = find_zero_flatliner(df, threshold)
        expected = pd.DataFrame(
            {
                "duration_h": [
                    (now - timedelta(minutes=15)) - (now - timedelta(minutes=45))
                ],
                "from_time": [now - timedelta(minutes=45)],
                "to_time": [now - timedelta(minutes=15)],
            }
        )
        self.assertDataframeEqual(
            expected, result, check_index_type=False, check_dtype=False
        )

    def test_some_NAN(self):
        """Data: a trafo contains a few NaN values and non compensated zero-value

        Expected: empty list, since there is no data to check
        """
        df = df_some_nan
        threshold = 0.25
        result = find_zero_flatliner(df, threshold)
        expected = pd.DataFrame(
            {
                "duration_h": [
                    (now - timedelta(minutes=15)) - (now - timedelta(minutes=45))
                ],
                "from_time": [now - timedelta(minutes=45)],
                "to_time": [now - timedelta(minutes=15)],
            }
        )
        self.assertDataframeEqual(
            expected, result, check_index_type=False, check_dtype=False
        )

    def test_compensated_flatliner(self):
        """Data: zero-values at one trafo is compensated

        Expected: empty list, since the zero-values are compensated and therefore reliable data (all trafo's)
        """
        df = df_compensated_flatliner
        threshold = 0.25
        result = find_zero_flatliner(df, threshold)
        expected = None
        self.assertEqual(result, expected)

    def test_no_data(self):
        """Data: empty database

        Expected: empty list, since there is no data to check
        """
        df = df_zero_file
        threshold = 0.25
        expected = None
        result = find_zero_flatliner(df, threshold)
        self.assertEqual(result, expected)

        # Run all tests

    def test_flatline_at_end(self):
        """The function should be able to handle arrays of unequal length. Bug KTPS-1823"""
        df = df_flatline_at_end
        threshold = 0.25
        result = find_zero_flatliner(df, threshold)
        self.assertEqual(result, None)

    def test_flatline_at_start(self):
        """The function should be able to handle arrays of unequal length. Bug KTPS-1823"""
        df = df_flatline_at_start
        threshold = 0.25
        result = find_zero_flatliner(df, threshold)
        self.assertEqual(result, None)


if __name__ == "__main__":
    unittest.main()
