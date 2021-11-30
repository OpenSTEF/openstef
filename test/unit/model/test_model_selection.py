# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np

from openstf.model_selection import model_selection

# define constants
SPLIT_PARAMS = {
    "test_fraction": 0.1,
    "validation_fraction": 0.15,
    "amount_day": 96,
}


class TestTrain(BaseTestCase):
    def test_sample_indices_train_val(self):
        """
        Test for sampling indices from a dataset.

        Raises:
            AssertionError: -

        """

        data = TestData.load("input_data_train.pickle")
        start_date = data.index.min().to_pydatetime()

        selection = [start_date]

        sampled, sampled_idx = model_selection.sample_indices_train_val(data, selection)
        self.assertEqual(len(selection), len(sampled))

    def test_random_sample(self):
        """
        Test for picking random samples from an np.array

        Raises:
            AssertionError: -

        """

        complete_list = np.array(7 * [10])
        n_random_samples = 4

        sampled_random = model_selection.random_sample(complete_list, n_random_samples)
        self.assertEqual(len(sampled_random), n_random_samples)

    def test_split_data_train_validation_test_stratification(self):

        """Test spliting data with stratification.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        data = TestData.load("input_data_train.pickle")

        (
            peaks,
            peaks_val_train,
            train_set,
            valid_set,
            test_set,
        ) = model_selection.split_data_train_validation_test(
            data,
            test_fraction=SPLIT_PARAMS["test_fraction"],
            validation_fraction=SPLIT_PARAMS["validation_fraction"],
        )

        # delta = 1, number of the peaks the two amounts may differ for the train and validation data
        # delta = 4, when looking at the test data, can differ 1 hr (4x15min)

        self.assertAlmostEqual(
            len(peaks_val_train[0][0]),
            len(peaks) * SPLIT_PARAMS["validation_fraction"],
            delta=1,
        )
        self.assertAlmostEqual(
            len(peaks_val_train[1][0]),
            len(peaks) * (1 - SPLIT_PARAMS["validation_fraction"]),
            delta=1,
        )

        self.assertAlmostEqual(
            len(test_set),
            len(data.index) * SPLIT_PARAMS["test_fraction"],
            delta=4,
        )


if __name__ == "__main__":
    unittest.main()
