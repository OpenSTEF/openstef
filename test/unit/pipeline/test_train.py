# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
import unittest

import pandas as pd

from test.utils.base import BaseTestCase
from test.utils.data import TestData

from openstf.pipeline.train_model import split_data_train_validation_test

# define constants
SPLIT_HYPER_PARAMS = {
    "test_fraction": 0.1,
    "validation_fraction": 0.15,
    "period_timedelta": 1,
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

    def test_split_data_train_validation_test_stratification(self):

        """Test spliting data with stratification.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        train_fraction = 1 - (
            SPLIT_HYPER_PARAMS["test_fraction"]
            + SPLIT_HYPER_PARAMS["validation_fraction"]
        )

        clean_data, train_set, valid_set, test_set = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_HYPER_PARAMS["test_fraction"],
            validation_fraction=SPLIT_HYPER_PARAMS["validation_fraction"],
            period_timedelta=SPLIT_HYPER_PARAMS["period_timedelta"],
            back_test=False,
            stratification=True,
        )

        self.assertAlmostEqual(
            len(valid_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["validation_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(test_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["test_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(train_set), len(clean_data.index) * train_fraction, delta=2 * 96
        )

    def test_split_data_train_validation_test_stratification_backtest(self):

        """Test spliting data with stratification and a backtest.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        train_fraction = 1 - (
            SPLIT_HYPER_PARAMS["test_fraction"]
            + SPLIT_HYPER_PARAMS["validation_fraction"]
        )

        clean_data, train_set, valid_set, test_set = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_HYPER_PARAMS["test_fraction"],
            validation_fraction=SPLIT_HYPER_PARAMS["validation_fraction"],
            period_timedelta=SPLIT_HYPER_PARAMS["period_timedelta"],
            back_test=True,
            stratification=True,
        )

        self.assertAlmostEqual(
            len(valid_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["validation_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(test_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["test_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(train_set), len(clean_data.index) * train_fraction, delta=2 * 96
        )

    def test_split_data_train_validation_test_no_stratification(self):

        """Test spliting data with no stratification.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        train_fraction = 1 - (
            SPLIT_HYPER_PARAMS["test_fraction"]
            + SPLIT_HYPER_PARAMS["validation_fraction"]
        )

        clean_data, train_set, valid_set, test_set = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_HYPER_PARAMS["test_fraction"],
            validation_fraction=SPLIT_HYPER_PARAMS["validation_fraction"],
            period_timedelta=SPLIT_HYPER_PARAMS["period_timedelta"],
            back_test=False,
            stratification=False,
        )

        self.assertAlmostEqual(
            len(valid_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["validation_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(test_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["test_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(train_set), len(clean_data.index) * train_fraction, delta=2 * 96
        )

    def test_split_data_train_validation_test_no_stratification_backtest(self):

        """Test spliting data with backtest and no stratification.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        train_fraction = 1 - (
            SPLIT_HYPER_PARAMS["test_fraction"]
            + SPLIT_HYPER_PARAMS["validation_fraction"]
        )

        clean_data, train_set, valid_set, test_set = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_HYPER_PARAMS["test_fraction"],
            validation_fraction=SPLIT_HYPER_PARAMS["validation_fraction"],
            period_timedelta=SPLIT_HYPER_PARAMS["period_timedelta"],
            back_test=True,
            stratification=False,
        )

        self.assertAlmostEqual(
            len(valid_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["validation_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(test_set),
            len(clean_data.index) * SPLIT_HYPER_PARAMS["test_fraction"],
            delta=2 * 96,
        )
        self.assertAlmostEqual(
            len(train_set), len(clean_data.index) * train_fraction, delta=2 * 96
        )


if __name__ == "__main__":
    unittest.main()
