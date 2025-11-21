# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <openstef@lfenergy.org> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import random
import unittest
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd

from openstef.model_selection import model_selection

# define constants
SPLIT_PARAMS = {
    "test_fraction": 0.1,
    "validation_fraction": 0.15,
    "amount_day": 96,
}


class TestTrain(BaseTestCase):
    def setUp(self) -> None:
        # seed random number generator so repeated tests yield same results
        np.random.seed(0)

    def test_sample_indices_train_val(self):
        """
        Test for sampling indices from a dataset.

        Raises:
            AssertionError: -

        """

        data = TestData.load("input_data_train.csv")
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

    def test_split_data_train_validation(self):
        """Test spliting data
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -
        """

        data = TestData.load("input_data_train.csv")

        (
            train_set,
            valid_set,
            test_set,
            operational_score_data,
        ) = model_selection.split_data_train_validation_test(
            data,
            test_fraction=SPLIT_PARAMS["test_fraction"],
            validation_fraction=SPLIT_PARAMS["validation_fraction"],
            stratification_min_max=False,
        )

        # delta = 1, number of the peaks the two amounts may differ for the train and validation data

        self.assertAlmostEqual(
            len(valid_set),
            len(data) * SPLIT_PARAMS["validation_fraction"],
            delta=2 * 96,
        )  # two days is allowed

        self.assertEqual(
            len(operational_score_data),
            len(data),
        )

    def test_split_data_train_validation_test_stratification(self):
        """
        Test that the train/validation split is stratified,
        meaning min/max days are equally distributed in train and validation set
        """
        # Arrange: prep inputs
        df = pd.DataFrame(
            index=pd.date_range(
                start="2021-01-01 00:00:00Z", freq="15T", periods=20 * 96
            )
        )
        # Make dates high/low
        # note that if this number does not match the validation_fraction,
        # the test results are not clearly defined (other dates can be recognized as min/max dates)
        n_days_high = 3
        n_days_low = 3
        # Specify load profile (each day is identical)
        df["load"] = np.sin(df.reset_index().index / 96 / 2 / np.pi * 20 * 2)
        # Randomly set max and min days
        max_days = random.sample(list(df.index.day.unique()), n_days_high)
        min_days = random.sample(
            [x for x in set(df.index.day) if x not in max_days], n_days_low
        )
        # Either increase or decrease load on those days
        for day in max_days:
            df.loc[df.index.day == day, "load"] += 5
        for day in min_days:
            df.loc[df.index.day == day, "load"] -= 5

        # Act: Split using default arguments. Should result in stratified split
        (
            train,
            val,
            test,
            operational_score_data,
        ) = model_selection.split_data_train_validation_test(
            df, test_fraction=0, stratification_min_max=True
        )

        # Assert: test that max and min days are both in train and val sets
        for dayset in [max_days, min_days]:
            for d in [train, val]:
                n_days_in = len([date for date in set(d.index.day) if date in dayset])

                # Add checkpoint, useful for debugging when things go wrong
                if n_days_in == 0:
                    print("Investigate this!")

                assert n_days_in >= 1

    def test_split_data_train_validation_test_stratification_repeat_20_times(self):
        """Repeat the stratification test 20 times,
        since the test using random selection and should work every time"""
        for _ in range(20):
            self.test_split_data_train_validation_test_stratification()


if __name__ == "__main__":
    unittest.main()
