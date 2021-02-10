# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

import numpy as np
import pandas as pd


class BaseTestCase(unittest.TestCase):

    # TODO for some reason this does not work : has not attribute addTypeEqualitty
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)
    #     self.addTypeEqualityFunc(pd.Series, self.assertSeriesEqual)
    #     self.addTypeEqualityFun(np.array, self.assertArrayEqual)

    def assertDataframeEqual(self, *args, **kwargs):
        try:
            pd.testing.assert_frame_equal(*args, **kwargs)
        except AssertionError as e:
            raise self.failureException from e

    def assertSeriesEqual(self, *args, **kwargs):
        try:
            pd.testing.assert_series_equal(*args, **kwargs)
        except AssertionError as e:
            raise self.failureException from e

    def assertArrayEqual(self, *args, **kwargs):
        try:
            np.testing.assert_array_equal(*args, **kwargs)
        except AssertionError as e:
            raise self.failureException from e

    def assertIsNAN(self, x):
        result = np.isnan(x)
        if type(result) is bool and result is False:
            raise self.failureException from AssertionError(
                f"x is not nan but '{type(x)}'"
            )
