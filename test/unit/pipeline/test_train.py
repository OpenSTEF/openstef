# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime, timedelta
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd

from openstef.pipeline.train_model import split_data_train_validation_test

# define constants
SPLIT_PARAMS = {
    "test_fraction": 0.1,
    "validation_fraction": 0.15,
    "amount_day": 96,
}


class TestTrain(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        datetime_start = datetime.utcnow() - timedelta(days=90)
        datetime_end = datetime.utcnow()
        self.data_table = TestData.load("input_data_train.pickle").head(8641)
        self.data = pd.DataFrame(
            index=pd.date_range(datetime_start, datetime_end, freq="15T")
        )
        # seed random number generator so repeated tests yield same results
        np.random.seed(0)

    def test_split_data_train_validation_test_stratification_no_backtest(self):
        """Test spliting data with stratification.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        (train_set, valid_set, test_set,) = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_PARAMS["test_fraction"],
            validation_fraction=SPLIT_PARAMS["validation_fraction"],
            back_test=False,
            stratification_min_max=True,
        )

        # delta = 1, number of the peaks the two amounts may differ for the train and validation data
        # delta = 4, when looking at the test data, can differ 1 hr (4x15min)
        self.assertAlmostEqual(
            len(set(valid_set.index.date)),
            len(set(self.data_table.index.date)) * SPLIT_PARAMS["validation_fraction"],
            delta=2,
        )

        self.assertAlmostEqual(
            len(test_set),
            len(self.data_table.index) * SPLIT_PARAMS["test_fraction"],
            delta=4,
        )

    def test_split_data_train_validation_test_stratification_backtest(self):
        """Test spliting data with stratification and a backtest.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        (train_set, valid_set, test_set,) = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_PARAMS["test_fraction"],
            validation_fraction=SPLIT_PARAMS["validation_fraction"],
            back_test=True,
            stratification_min_max=True,
        )

        # delta = 1, number of the peaks the two amounts may differ for the train and validation data
        # delta = 4, when looking at the test data, can differ 1 hr (4x15min)

        self.assertAlmostEqual(
            len(set(valid_set.index.date)),
            len(set(self.data_table.index.date)) * SPLIT_PARAMS["validation_fraction"],
            delta=2,
        )

        self.assertAlmostEqual(
            len(test_set),
            len(self.data_table.index) * SPLIT_PARAMS["test_fraction"],
            delta=4,
        )

    def test_split_data_train_validation_test_no_stratification_no_backtest(self):
        """Test spliting data with no stratification.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        train_fraction = 1 - (
            SPLIT_PARAMS["test_fraction"] + SPLIT_PARAMS["validation_fraction"]
        )

        (train_set, valid_set, test_set,) = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_PARAMS["test_fraction"],
            validation_fraction=SPLIT_PARAMS["validation_fraction"],
            back_test=False,
            stratification_min_max=False,
        )

        # delta = 4, when looking at the test data, can differ 1 hr (4x15min)

        self.assertAlmostEqual(
            len(test_set),
            len(self.data_table.index) * SPLIT_PARAMS["test_fraction"],
            delta=4,
        )
        self.assertAlmostEqual(
            len(set(valid_set.index.date)),
            len(set(self.data_table.index.date)) * SPLIT_PARAMS["validation_fraction"],
            delta=2,
        )
        self.assertAlmostEqual(
            len(train_set),
            len(self.data_table.index) * train_fraction,
            delta=4,
        )

    def test_split_data_train_validation_test_no_stratification_backtest(self):
        """Test spliting data with backtest and no stratification.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        train_fraction = 1 - (
            SPLIT_PARAMS["test_fraction"] + SPLIT_PARAMS["validation_fraction"]
        )

        (train_set, valid_set, test_set,) = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_PARAMS["test_fraction"],
            validation_fraction=SPLIT_PARAMS["validation_fraction"],
            back_test=True,
            stratification_min_max=False,
        )

        # delta = 4, when looking at the test data, can differ 1 hr (4x15min)

        self.assertAlmostEqual(
            len(test_set),
            len(self.data_table.index) * SPLIT_PARAMS["test_fraction"],
            delta=4,
        )
        self.assertAlmostEqual(
            len(set(valid_set.index.date)),
            len(set(self.data_table.index.date)) * SPLIT_PARAMS["validation_fraction"],
            delta=2,
        )
        self.assertAlmostEqual(
            len(train_set),
            len(self.data_table.index) * train_fraction,
            delta=4,
        )


if __name__ == "__main__":
    unittest.main()
