# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from numpy import split

from numpy.core.fromnumeric import mean
from test.utils import BaseTestCase, TestData

from openstf.model import split_energy

# Get test data
input_data = TestData.load("find_components_input.csv")
components = TestData.load("find_components_components.csv")


class TestSplitEnergy(BaseTestCase):
    def test_find_components(self):
        testcomponents, coefdict = split_energy.find_components(input_data)
        self.assertDataframeEqual(components, testcomponents, rtol=1e-3)

    threshold = split_energy.COEF_MAX_PCT_DIFF

    def test_are_coefs_valid_true(self):
        new_coefs = {"a": 1, "b": 1, "c": -1}
        mean_coefs = {"a": 1, "b": 1 + self.threshold, "c": -1}

        result = split_energy.are_new_coefs_valid(new_coefs, mean_coefs)
        self.assertTrue(result)

    def test_are_coefs_valid_flipped_sign(self):
        new_coefs = {"a": 1, "b": 1}
        mean_coefs = {"a": 1, "b": -1}

        result = split_energy.are_new_coefs_valid(new_coefs, mean_coefs)
        self.assertFalse(result)

    def test_are_coefs_valid_above_threshold(self):
        new_coefs = {"a": 1, "b": 1}
        mean_coefs = {"a": 1, "b": 1 + 1.5 * self.threshold}

        result = split_energy.are_new_coefs_valid(new_coefs, mean_coefs)
        self.assertFalse(result)

    def test_are_coefs_valid_below_threshold(self):
        new_coefs = {"a": 1, "b": 1}
        mean_coefs = {"a": 1, "b": 1 - self.threshold}

        result = split_energy.are_new_coefs_valid(new_coefs, mean_coefs)
        self.assertFalse(result)


# Run all tests
if __name__ == "__main__":
    unittest.main()
