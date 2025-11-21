# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <openstef@lfenergy.org> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

import numpy as np
import pandas as pd


class dotdict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)
        self.addTypeEqualityFunc(pd.Series, self.assertSeriesEqual)
        self.addTypeEqualityFunc(np.array, self.assertArrayEqual)

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
