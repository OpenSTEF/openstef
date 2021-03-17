# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from openstf.model.validation_robot import nonzero_flatliner, replace_invalid_data

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


class ReplaceInvalidDataTest(BaseTestCase):
    def test_no_flatliner(self):
        df = df_no_flatliner
        suspicious_moments = nonzero_flatliner(df, 0.25)
        expected = df
        result = replace_invalid_data(df, suspicious_moments)

        self.assertDataframeEqual(expected, result)

    def test_station_flatliner(self):
        """Data: all trafo's containing non zero-value flatliners

        Expected: list containing all trafo values and timestamps of the stationsflatliners + the generates diff_columns to detect the flatliners.
        """
        df = df_flatliner
        suspicious_moments = nonzero_flatliner(df, 0.25)
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
        result = replace_invalid_data(df, suspicious_moments)
        self.assertDataframeEqual(expected, result)

    def test_no_data(self):
        """Data: empty database

        Expected: empty list, since there is no data to check
        """
        df = df_zero_file
        suspicious_moments = nonzero_flatliner(df, 0.25)
        expected = df
        result = replace_invalid_data(df, suspicious_moments)
        self.assertDataframeEqual(expected, result)

    # Run all tests


if __name__ == "__main__":
    unittest.main()
