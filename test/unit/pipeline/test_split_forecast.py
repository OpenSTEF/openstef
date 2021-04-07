# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
import pandas as pd
import numpy as np

from test.utils import BaseTestCase, TestData

from openstf.pipeline import split_forecast

# Get test data
input_data = TestData.load("find_components_input.csv")
components = TestData.load("find_components_components.csv")


class TestSplitEnergy(BaseTestCase):
    def test_find_components(self):
        testcomponents, coefdict = split_forecast.find_components(input_data)
        self.assertDataframeEqual(components, testcomponents, rtol=1e-3)

    threshold = split_forecast.COEF_MAX_FRACTION_DIFF

    def test_determine_invalid_coefs_valid(self):
        new_coefs = pd.DataFrame(
            {"coef_name": ["a", "b", "c"], "coef_value": [1, 1, -1]}
        )
        last_coefs = pd.DataFrame(
            {"coef_name": ["a", "b", "c"], "coef_value": [1, 1 + self.threshold, -1]}
        )

        expected_result = pd.DataFrame(
            columns=[
                "coef_name",
                "coef_value_last",
                "coef_value_new",
                "difference",
            ],
        )
        result = split_forecast.determine_invalid_coefs(new_coefs, last_coefs)
        self.assertDataframeEqual(
            result, expected_result, check_dtype=False, check_index_type=False
        )

    def test_determine_invalid_coefs_flipped_sign(self):
        new_coefs = pd.DataFrame({"coef_name": ["a", "b"], "coef_value": [1, 1]})
        last_coefs = pd.DataFrame({"coef_name": ["a", "b"], "coef_value": [1, -1]})

        expected_result = pd.DataFrame(
            {
                "coef_name": ["b"],
                "coef_value_last": [-1],
                "coef_value_new": [1],
                "difference": [2],
            },
            index=[1],
        )
        result = split_forecast.determine_invalid_coefs(new_coefs, last_coefs)
        self.assertDataframeEqual(result, expected_result, check_index_type=False)

    def test_determine_invalid_coefs_above_threshold(self):
        new_coefs = pd.DataFrame({"coef_name": ["a", "b"], "coef_value": [1, 1]})
        last_coefs = pd.DataFrame(
            {"coef_name": ["a", "b"], "coef_value": [1, 1 + 1.5 * self.threshold]}
        )

        expected_result = pd.DataFrame(
            {
                "coef_name": ["b"],
                "coef_value_last": [1 + 1.5 * self.threshold],
                "coef_value_new": [1],
                "difference": [1.5 * self.threshold],
            },
            index=[1],
        )
        result = split_forecast.determine_invalid_coefs(new_coefs, last_coefs)
        self.assertDataframeEqual(result, expected_result, check_index_type=False)

    def test_determine_invalid_coefs_below_threshold(self):
        new_coefs = pd.DataFrame({"coef_name": ["a", "b"], "coef_value": [1, 1]})
        last_coefs = pd.DataFrame(
            {"coef_name": ["a", "b"], "coef_value": [1, 1 - self.threshold]}
        )

        expected_result = pd.DataFrame(
            {
                "coef_name": ["b"],
                "coef_value_last": [1 - self.threshold],
                "coef_value_new": [1],
                "difference": [self.threshold],
            },
            index=[1],
        )
        result = split_forecast.determine_invalid_coefs(new_coefs, last_coefs)
        self.assertDataframeEqual(result, expected_result, check_index_type=False)

    def test_determine_invalid_coefs_multiple_failing_keys(self):
        new_coefs = pd.DataFrame(
            {"coef_name": ["a", "b", "c"], "coef_value": [1, 1, 1]}
        )
        last_coefs = pd.DataFrame(
            {"coef_name": ["a", "b", "c"], "coef_value": [1, 1 - self.threshold, -1]}
        )

        expected_result = pd.DataFrame(
            {
                "coef_name": ["b", "c"],
                "coef_value_last": [1 - self.threshold, -1],
                "coef_value_new": [1, 1],
                "difference": [self.threshold, 2],
            },
            index=[1, 2],
        )
        result = split_forecast.determine_invalid_coefs(new_coefs, last_coefs)
        self.assertDataframeEqual(result, expected_result, check_index_type=False)

    def test_determine_invalid_coefs_no_matching_key(self):
        new_coefs = pd.DataFrame({"coef_name": ["a", "b"], "coef_value": [1, 1]})
        last_coefs = pd.DataFrame({"coef_name": ["a", "c"], "coef_value": [1, 1]})

        expected_result = pd.DataFrame(
            {
                "coef_name": ["c"],
                "coef_value_last": [1],
                "coef_value_new": [np.nan],
                "difference": [np.inf],
            },
            index=[1],
        )
        result = split_forecast.determine_invalid_coefs(new_coefs, last_coefs)
        self.assertDataframeEqual(result, expected_result, check_index_type=False)


# Run all tests
if __name__ == "__main__":
    unittest.main()
