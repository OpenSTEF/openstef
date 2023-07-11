# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
from test.unit.utils.base import BaseTestCase

import pandas as pd

from openstef.validation.validation import check_data_for_each_trafo

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
df_nan_value = pd.DataFrame(
    {
        "date": date_rng2,
        "col1": [8.0, 8.0, 8.0, 8.0, 8.0],
        "col2": [None, None, None, None, None],
    }
)
df_zero_file = pd.DataFrame({"date5": [], "col1": [], "col2": []})
df_all_zerovalues = pd.DataFrame(
    {"date6": date_rng, "col1": [8.16, 7.64, 7.44], "col2": [0, 0, 0]}
)

df_flatliner = df_flatliner.set_index("date")
df_no_flatliner = df_no_flatliner.set_index("date")
df_nan_value = df_nan_value.set_index("date")
df_zero_file = df_zero_file.set_index("date5")
df_all_zerovalues = df_all_zerovalues.set_index("date6")


class TestValidationCheckDataForEachTrafo(BaseTestCase):
    def test_no_zero_values(self):
        """Data: no zero-values in any of the trafo's

        Expected: False, since there is not even one zero-value
        """
        df = df_no_flatliner
        col = "col1"
        expected = False
        result = check_data_for_each_trafo(df, col)

        self.assertEqual(expected, result)

    def test_all_zerovalues(self):
        """Data: one trafo containing all zero_values

        Expected: False, since all data is zero so probably in construction
        """
        df = df_all_zerovalues
        col = "col2"
        result = check_data_for_each_trafo(df, col)
        expected = False
        self.assertEqual(result, expected)

    def test_all_NAN(self):
        """Data: one trafo containing NaN values

        Expected: False, since there is no data
        """
        df = df_nan_value
        col = "col2"
        result = check_data_for_each_trafo(df, col)
        expected = False
        self.assertEqual(result, expected)

    def test_no_data(self):
        """Data: empty database

        Expected: False, since there is no data
        """
        df = df_zero_file
        col = "col2"
        expected = False
        result = check_data_for_each_trafo(df, col)
        self.assertEqual(result, expected)

    def test_some_zero_values(self):
        """Data: some zero-values in any of the trafo's

        Expected: True, since there are zero-values to check
        """
        df = df_flatliner
        col = "col2"
        expected = True
        result = check_data_for_each_trafo(df, col)

        self.assertEqual(expected, result)
