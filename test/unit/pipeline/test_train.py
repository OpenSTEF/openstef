# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from datetime import datetime, timedelta, UTC
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from openstef.data_classes.split_function import SplitFuncDataClass
from openstef.model_selection.model_selection import split_data_train_validation_test
from openstef.pipeline.train_model import train_pipeline_step_split_data

# define constants
SPLIT_PARAMS = {
    "test_fraction": 0.1,
    "validation_fraction": 0.15,
    "amount_day": 96,
}


def dummy_split(data, test_fraction):
    return data.iloc[:100], data.iloc[100:110], data.iloc[110:120], data


def sk_split(data, test_fraction, gap):
    tscv = TimeSeriesSplit(
        n_splits=2, test_size=int(np.round(test_fraction * len(data))), gap=gap
    )
    splits = list(tscv.split(data))
    return (
        data.iloc[splits[0][0]],
        data.iloc[splits[0][1]],
        data.iloc[splits[1][1]],
        data,
    )


class TestTrain(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        datetime_start = datetime.now(tz=UTC) - timedelta(days=90)
        datetime_end = datetime.now(tz=UTC)
        self.data_table = TestData.load("input_data_train.csv").head(8641)
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

        (
            train_set,
            valid_set,
            test_set,
            operational_score_data,
        ) = split_data_train_validation_test(
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

        self.assertEqual(
            len(operational_score_data),
            len(self.data_table.index),
        )

    def test_split_data_train_validation_test_stratification_backtest(self):
        """Test spliting data with stratification and a backtest.
            Test the `split_data_stratification` function and compare the proportion of the split
            of data into training, test, and validation subsets with the fractions.

        Raises:
            AssertionError: -

        """

        (
            train_set,
            valid_set,
            test_set,
            operational_score_data,
        ) = split_data_train_validation_test(
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

        train_fraction = 1 - SPLIT_PARAMS["validation_fraction"]

        (
            train_set,
            valid_set,
            test_set,
            operational_score_data,
        ) = split_data_train_validation_test(
            self.data_table,
            test_fraction=SPLIT_PARAMS["test_fraction"],
            validation_fraction=SPLIT_PARAMS["validation_fraction"],
            back_test=False,
            stratification_min_max=False,
        )

        # delta = 4, when looking at the test data, can differ 1 hr (4x15min)

        self.assertEqual(
            len(operational_score_data),
            len(self.data_table.index),
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

        (
            train_set,
            valid_set,
            test_set,
            operational_score_data,
        ) = split_data_train_validation_test(
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

    def test_train_pipeline_step_split_data(self):
        (
            train_set,
            valid_set,
            test_set,
            operational_score_data,
        ) = train_pipeline_step_split_data(
            self.data_table,
            self.pj,
            test_fraction=SPLIT_PARAMS["test_fraction"],
        )

        self.assertEqual(
            len(operational_score_data),
            len(self.data_table.index),
        )
        self.assertAlmostEqual(
            len(set(valid_set.index.date)),
            len(set(self.data_table.index.date)) * SPLIT_PARAMS["validation_fraction"],
            delta=2,
        )

    def test_train_pipeline_step_split_data_test_data_predefined(self):
        test_data_predefined = self.data_table.tail(15)
        (
            train_data,
            validation_data,
            test_data,
            operational_score_data,
        ) = train_pipeline_step_split_data(
            self.data_table,
            self.pj,
            test_fraction=0,
            test_data_predefined=test_data_predefined,
        )

        self.assertTrue(test_data.equals(test_data_predefined))

    def test_train_pipeline_step_split_data_custom_split(self):
        pj = self.pj

        # Test wrong custom split
        pj.train_split_func = SplitFuncDataClass(function="unkown_split", arguments={})
        with self.assertRaises(ValueError):
            (
                train_data,
                validation_data,
                test_data,
                operational_score_data,
            ) = train_pipeline_step_split_data(
                self.data_table,
                pj,
                test_fraction=0,
            )

        pj.train_split_func = SplitFuncDataClass(
            function=lambda data: dummy_split(data, 0), arguments={}
        )
        with self.assertRaises(ValueError):
            (
                train_data,
                validation_data,
                test_data,
                operational_score_data,
            ) = train_pipeline_step_split_data(
                self.data_table,
                pj,
                test_fraction=0,
            )

        # Test dummy custom split
        pj.train_split_func = SplitFuncDataClass(function=dummy_split, arguments={})
        (
            train_data,
            validation_data,
            test_data,
            operational_score_data,
        ) = train_pipeline_step_split_data(
            self.data_table,
            pj,
            test_fraction=0,
        )
        self.assertTrue(train_data.equals(self.data_table.iloc[:100]))
        self.assertTrue(validation_data.equals(self.data_table.iloc[100:110]))
        self.assertTrue(test_data.equals(self.data_table.iloc[110:120]))

        # Test dummy custom split as json
        pj.train_split_func = SplitFuncDataClass(
            function="test.unit.pipeline.test_train.dummy_split", arguments="{}"
        )
        (
            train_data,
            validation_data,
            test_data,
            operational_score_data,
        ) = train_pipeline_step_split_data(
            self.data_table,
            pj,
            test_fraction=0,
        )
        self.assertTrue(train_data.equals(self.data_table.iloc[:100]))
        self.assertTrue(validation_data.equals(self.data_table.iloc[100:110]))
        self.assertTrue(test_data.equals(self.data_table.iloc[110:120]))

        # Test sklearn custom split with argument
        pj.train_split_func = SplitFuncDataClass(
            function=sk_split, arguments={"gap": 10}
        )
        (
            train_data,
            validation_data,
            test_data,
            operational_score_data,
        ) = train_pipeline_step_split_data(
            self.data_table,
            pj,
            test_fraction=SPLIT_PARAMS["test_fraction"],
        )
        self.assertAlmostEqual(
            len(test_data),
            len(self.data_table.index) * SPLIT_PARAMS["test_fraction"],
            delta=4,
        )


if __name__ == "__main__":
    unittest.main()
