# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import math
import unittest

import numpy as np
import pandas as pd

from stf.feature_engineering.general import nan_repeated

from test.utils import BaseTestCase


class GeneralTest(BaseTestCase):

    def test_nan_repeated(self):
        start_nan = 8
        end_nan = 11
        data = np.arange(20)
        df = pd.DataFrame({"Column1": data[:], "Column2": data[:], "Column3": data[:]})
        # Add 4 sequentially same numbers
        df["Column2"] = np.where(
            df["Column2"].between(start_nan, end_nan), 30, df["Column2"]
        )

        # No change, since there are 4 seq. numbers.
        df_no_repeated = nan_repeated(df, 5, "Column2")
        print(df_no_repeated.isna().values.any())
        self.assertFalse(df_no_repeated.isna().values.any())

        # No change, since there are 4 seq. numbers.
        df_no_repeated = nan_repeated(df, 4, "Column2")
        self.assertFalse(df_no_repeated.isna().values.any())

        # Change, since there are 4 seq. numbers and we don't allow for more than 3
        df_no_repeated = nan_repeated(df, 3, "Column2")
        self.assertEqual(df_no_repeated["Column1"].isna().values.sum(), 0)
        self.assertEqual(df_no_repeated["Column2"].isna().values.sum(), 1)
        self.assertEqual(df_no_repeated["Column3"].isna().values.sum(), 0)
        self.assertTrue(df_no_repeated["Column1"].equals(df_no_repeated["Column3"]))
        self.assertTrue(math.isnan(df_no_repeated.at[end_nan, "Column2"]))

        # Change, since there are 4 seq. numbers and we don't allow for more than 2
        df_no_repeated = nan_repeated(df, 2, "Column2")
        self.assertEqual(df_no_repeated["Column1"].isna().values.sum(), 0)
        self.assertEqual(df_no_repeated["Column2"].isna().values.sum(), 2)
        self.assertEqual(df_no_repeated["Column3"].isna().values.sum(), 0)
        self.assertTrue(df_no_repeated["Column1"].equals(df_no_repeated["Column3"]))
        self.assertTrue(math.isnan(df_no_repeated.at[end_nan, "Column2"]))
        self.assertTrue(math.isnan(df_no_repeated.at[end_nan - 1, "Column2"]))
