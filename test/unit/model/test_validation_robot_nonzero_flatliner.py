# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import unittest
from datetime import datetime, timedelta

import pandas as pd

from stf.model.validation_robot import nonzero_flatliner

from test.utils import BaseTestCase

now = datetime.utcnow()
date_rng = pd.date_range(start=now - timedelta(minutes=30), end=now, freq="0.25H")
date_rng2 = pd.date_range(start=now - timedelta(minutes=60), end=now, freq="0.25H")
date_rng3 = pd.date_range(start=now - timedelta(minutes=30), end=now, freq="0.25H")

df_no_flatliner = pd.DataFrame(
    {
        "date": date_rng,
        "LC_trafo": [0, 0, 0],
        "col1": [8.16, 7.64, 7.44],
        "col2": [0, 0, 0],
    }
)
df_flatliner = pd.DataFrame(
    {
        "date": date_rng2,
        "LC_trafo": [0, 0, 0, 0, 0],
        "col1": [8.0, 8.16, 8.16, 8.16, 8.0],
        "col2": [8.0, 8.75, 8.75, 8.75, 8.0],
    }
)
df_zerovalues_flatliner = pd.DataFrame(
    {
        "date": date_rng2,
        "LC_trafo": [0, 0, 0, 0, 0],
        "col1": [8.0, 8.16, 8.16, 8.16, 8.0],
        "col2": [8.0, 0, 0, 0, 8.0],
    }
)
df_trafo_flatliner = pd.DataFrame(
    {
        "date4": date_rng,
        "LC_trafo": [0, 0, 0],
        "col1": [8.16, 8.16, 8.16],
        "col2": [8.16, 7.64, 7.44],
    }
)
df_zero_file = pd.DataFrame({"date5": [], "LC_trafo": [], "col1": [], "col2": []})
df_zero_flatliner = pd.DataFrame(
    {"date6": date_rng, "LC_trafo": [0, 0, 0], "col1": [0, 0, 0], "col2": [0, 0, 0]}
)


df_no_flatliner = df_no_flatliner.set_index("date")
df_flatliner = df_flatliner.set_index("date")
df_zerovalues_flatliner = df_zerovalues_flatliner.set_index("date")
df_trafo_flatliner = df_trafo_flatliner.set_index("date4")
df_zero_file = df_zero_file.set_index("date5")
df_zero_flatliner = df_zero_flatliner.set_index("date6")


# eerste column invoegen (die weer verwijderd wordt)
# @mock.patch("stf.model.nonzero_flatliner_detection.query_load_data", return_value = df_no_flatliner)
class NonZeroFlatlinerTest(BaseTestCase):
    def test_no_flatliner(self):
        df = df_no_flatliner
        threshold = 0.25
        expected = None
        result = nonzero_flatliner(df, threshold)

        self.assertEqual(result, expected)

    def test_station_flatliner(self):
        """Data: all trafo's containing non zero-value flatliners

        Expected: list containing all trafo values and timestamps of the stationsflatliners + the generates diff_columns to detect the flatliners.
        """
        df = df_flatliner
        threshold = 0.25
        # create expected output with diff_columns
        df_flatliner_output = pd.DataFrame(
            {
                "from_time": [now - timedelta(minutes=45)],
                "to_time": [now - timedelta(minutes=15)],
                "duration_h": [
                    (now - timedelta(minutes=15)) - (now - timedelta(minutes=45))
                ],
            }
        )
        expected = []
        expected.append(df_flatliner_output)
        result = nonzero_flatliner(df, threshold)
        expected_output = expected[0]
        self.assertDataframeEqual(
            expected_output, result, check_index_type=False, check_dtype=False
        )

    def test_flatliner_zerovalues(self):
        """Data: one trafo showing a non zero-value flatliner, one trafo showing a zero-value flatliner

        Expected: list containing all trafo values and timestamps of the stationsflatliners + the generates diff_columns to detect the flatliners.
        """
        df = df_zerovalues_flatliner
        threshold = 0.25
        # create expected output with diff_columns
        df_zerovalues_flatliner_output = pd.DataFrame(
            {
                "from_time": [now - timedelta(minutes=45)],
                "to_time": [now - timedelta(minutes=15)],
                "duration_h": [
                    (now - timedelta(minutes=15)) - (now - timedelta(minutes=45))
                ],
            }
        )
        expected = []
        expected.append(df_zerovalues_flatliner_output)
        result = nonzero_flatliner(df, threshold)
        expected_output = expected[0]
        self.assertDataframeEqual(
            expected_output, result, check_index_type=False, check_dtype=False
        )

    def test_trafo_flatliner(self):
        """Data: only one trafo shows a flatliner

        Expected: empty list, since the function only detects a complete station flatliner (all trafo's)
        """
        df = df_trafo_flatliner
        threshold = 0.25
        expected = None
        result = nonzero_flatliner(df, threshold)
        self.assertEqual(result, expected)

    def test_no_data(self):
        """Data: empty database

        Expected: empty list, since there is no data to check
        """
        df = df_zero_file
        threshold = 0.25
        expected = None
        result = nonzero_flatliner(df, threshold)
        self.assertEqual(result, expected)

    def test_zero_flatliner(self):
        """Data: zero value flatliner for complete station

        Expected: empty list, since the function detects only non-zero station flatliners
        """
        df = df_zero_flatliner
        threshold = 0.25
        expected = None
        result = nonzero_flatliner(df, threshold)
        self.assertEqual(result, expected)

    # Run all tests


if __name__ == "__main__":
    unittest.main()
