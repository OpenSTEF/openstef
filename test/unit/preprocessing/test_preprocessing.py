# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import math
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from openstf.preprocessing import preprocessing
from openstf.validation import validation
from test.utils import BaseTestCase

now = datetime.utcnow()
date_range = pd.date_range(start=now - timedelta(minutes=30), end=now, freq="0.25H")
date_range2 = pd.date_range(start=now - timedelta(minutes=60), end=now, freq="0.25H")
date_range3 = pd.date_range(start=now - timedelta(minutes=30), end=now, freq="0.25H")

df_no_flatliner = pd.DataFrame(
    {
        "date": date_range,
        "LC_trafo": [0, 0, 0],
        "col1": [8.16, 7.64, 7.44],
        "col2": [0, 0, 0],
    }
)
df_flatliner = pd.DataFrame(
    {
        "date": date_range2,
        "LC_trafo": [0, 0, 0, 0, 0],
        "col1": [8.0, 8.16, 8.16, 8.16, 8.0],
        "col2": [8.0, 8.75, 8.75, 8.75, 8.0],
    }
)
df_zero_file = pd.DataFrame({"date5": [], "LC_trafo": [], "col1": [], "col2": []})


df_no_flatliner = df_no_flatliner.set_index("date")
df_flatliner = df_flatliner.set_index("date")
df_zero_file = df_zero_file.set_index("date5")


class TestPreprocessing(BaseTestCase):
    def test_replace_repeated_values_with_nan(self):
        start_nan = 8
        end_nan = 11
        data = np.arange(20)
        df = pd.DataFrame({"Column1": data[:], "Column2": data[:], "Column3": data[:]})
        # Add 4 sequentially same numbers
        df["Column2"] = np.where(
            df["Column2"].between(start_nan, end_nan), 30, df["Column2"]
        )

        # No change, since there are 4 seq. numbers.
        df_no_repeated = preprocessing.replace_repeated_values_with_nan(
            df, 5, "Column2"
        )
        print(df_no_repeated.isna().values.any())
        self.assertFalse(df_no_repeated.isna().values.any())

        # No change, since there are 4 seq. numbers.
        df_no_repeated = preprocessing.replace_repeated_values_with_nan(
            df, 4, "Column2"
        )
        self.assertFalse(df_no_repeated.isna().values.any())

        # Change, since there are 4 seq. numbers and we don't allow for more than 3
        df_no_repeated = preprocessing.replace_repeated_values_with_nan(
            df, 3, "Column2"
        )
        self.assertEqual(df_no_repeated["Column1"].isna().values.sum(), 0)
        self.assertEqual(df_no_repeated["Column2"].isna().values.sum(), 1)
        self.assertEqual(df_no_repeated["Column3"].isna().values.sum(), 0)
        self.assertTrue(df_no_repeated["Column1"].equals(df_no_repeated["Column3"]))
        self.assertTrue(math.isnan(df_no_repeated.at[end_nan, "Column2"]))

        # Change, since there are 4 seq. numbers and we don't allow for more than 2
        df_no_repeated = preprocessing.replace_repeated_values_with_nan(
            df, 2, "Column2"
        )
        self.assertEqual(df_no_repeated["Column1"].isna().values.sum(), 0)
        self.assertEqual(df_no_repeated["Column2"].isna().values.sum(), 2)
        self.assertEqual(df_no_repeated["Column3"].isna().values.sum(), 0)
        self.assertTrue(df_no_repeated["Column1"].equals(df_no_repeated["Column3"]))
        self.assertTrue(math.isnan(df_no_repeated.at[end_nan, "Column2"]))
        self.assertTrue(math.isnan(df_no_repeated.at[end_nan - 1, "Column2"]))

    def test_no_flatliner(self):
        df = df_no_flatliner
        suspicious_moments = validation.find_nonzero_flatliner(df, 0.25)

        expected = df
        result = preprocessing.replace_invalid_data(df, suspicious_moments)

        self.assertDataframeEqual(expected, result)

    def test_station_flatliner(self):
        """Data: all trafo's containing non zero-value flatliners

        Expected: list containing all trafo values and timestamps of the
        stationsflatliners + the generates diff_columns to detect the flatliners.
        """
        df = df_flatliner
        suspicious_moments = validation.find_nonzero_flatliner(df, 0.25)
        # create expected output with diff_columns
        df_flatliner_output = pd.DataFrame(
            {
                "date": date_range2,
                "LC_trafo": [0, np.nan, np.nan, np.nan, 0],
                "col1": [8.0, np.nan, np.nan, np.nan, 8.0],
                "col2": [8.0, np.nan, np.nan, np.nan, 8.0],
            }
        )
        df_flatliner_output = df_flatliner_output.set_index("date")
        expected = df_flatliner_output
        result = preprocessing.replace_invalid_data(df, suspicious_moments)
        self.assertDataframeEqual(expected, result)

    def test_no_data(self):
        """Data: empty database

        Expected: empty list, since there is no data to check
        """
        df = df_zero_file
        suspicious_moments = validation.find_nonzero_flatliner(df, 0.25)
        expected = df
        result = preprocessing.replace_invalid_data(df, suspicious_moments)
        self.assertDataframeEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
