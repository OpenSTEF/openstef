# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

import numpy as np
import pandas as pd

from stf.feature_engineering import apply_features

from test.utils import BaseTestCase, TestData


class TestApplyFeaturesAditionalMinuteSpace(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.data = TestData.load("input_data_train.pickle")

    def test_additional_minutes_space(self):
        minute_space_list = apply_features.additional_minute_space(self.data)
        self.assertEqual(minute_space_list, [1410, 2880])

    def test_empty_data_frame(self):
        minute_space_list = apply_features.additional_minute_space(pd.DataFrame())
        self.assertEqual(minute_space_list, [])

    def test_no_peaks_in_correlation(self):
        data = pd.DataFrame()
        data["random_column_name"] = np.linspace(0, 1000, 1000, endpoint=False)
        minute_space_list = apply_features.additional_minute_space(data)
        self.assertEqual(minute_space_list, [])


# Run all tests
if __name__ == "__main__":
    unittest.main()
