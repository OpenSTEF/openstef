# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from openstef.data_classes.split_function import SplitFuncDataClass
from openstef.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstef.pipeline.train_create_forecast_backtest import (
    train_model_and_forecast_back_test,
)
from openstef.validation import validation


def timeseries_split(data, n_folds, gap):
    test_fraction = 0.15
    nb_test = int(np.round(test_fraction * len(data)))
    test_set = data.iloc[-nb_test:]
    input_set = data.iloc[:-nb_test]
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=nb_test, gap=gap)
    return [
        (input_set.iloc[idx_train], input_set.iloc[idx_val], test_set)
        for idx_train, idx_val in tscv.split(input_set)
    ]


class TestTrainBackTestPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    def test_train_model_pipeline_core_happy_flow(self):
        """Test happy flow of the train model pipeline"""

        (
            forecast,
            model,
            train_data,
            validation_data,
            test_data,
        ) = train_model_and_forecast_back_test(
            pj=self.pj,
            modelspecs=self.modelspecs,
            input_data=self.train_input,
            training_horizons=[0.25, 24.0],
        )

        self.assertTrue("forecast" in forecast.columns)
        self.assertTrue("realised" in forecast.columns)
        self.assertTrue("horizon" in forecast.columns)
        self.assertEqual(set(forecast.horizon.unique()), {0.25, 24.0})

    def test_train_model_pipeline_core_happy_flow_nfold(self):
        """Test happy flow of the train model pipeline, using cross validation to forecast the entire input range"""

        (
            forecast,
            model,
            train_data,
            validation_data,
            test_data,
        ) = train_model_and_forecast_back_test(
            pj=self.pj,
            modelspecs=self.modelspecs,
            input_data=self.train_input,
            training_horizons=[0.25, 24.0],
            n_folds=4,
        )

        self.assertTrue("forecast" in forecast.columns)
        self.assertTrue("realised" in forecast.columns)
        self.assertTrue("horizon" in forecast.columns)
        self.assertEqual(sorted(list(forecast.horizon.unique())), [0.25, 24.0])

        # check if forecast is indeed of the entire range of the input data
        validated_data = validation.drop_target_na(
            validation.validate(
                self.pj["id"], self.train_input, self.pj["flatliner_treshold"]
            )
        )
        data_with_features = TrainFeatureApplicator(
            horizons=[0.25, 24.0], feature_names=self.modelspecs.feature_names
        ).add_features(validated_data, pj=self.pj)
        self.assertEqual(len(forecast), len(data_with_features))

    def test_train_model_pipeline_core_custom_split(self):

        pj = self.pj
        # test wrong custom backtest split
        pj.backtest_split_func = SplitFuncDataClass(
            function="unknow_backtest", arguments={}
        )
        with self.assertRaises(ValueError):
            (
                forecast,
                model,
                train_data,
                validation_data,
                test_data,
            ) = train_model_and_forecast_back_test(
                pj=self.pj,
                modelspecs=self.modelspecs,
                input_data=self.train_input,
                training_horizons=[0.25, 24.0],
                n_folds=4,
            )

        pj.backtest_split_func = SplitFuncDataClass(
            function=lambda data: timeseries_split(data, 0, 24), arguments={}
        )
        with self.assertRaises(ValueError):
            (
                forecast,
                model,
                train_data,
                validation_data,
                test_data,
            ) = train_model_and_forecast_back_test(
                pj=self.pj,
                modelspecs=self.modelspecs,
                input_data=self.train_input,
                training_horizons=[0.25, 24.0],
                n_folds=4,
            )

        # test custom backtest split
        pj.backtest_split_func = SplitFuncDataClass(
            function=timeseries_split, arguments={"gap": 24}
        )
        (
            forecast,
            model,
            train_data,
            validation_data,
            test_data,
        ) = train_model_and_forecast_back_test(
            pj=self.pj,
            modelspecs=self.modelspecs,
            input_data=self.train_input,
            training_horizons=[0.25, 24.0],
            n_folds=4,
        )

        self.assertTrue("forecast" in forecast.columns)
        self.assertTrue("realised" in forecast.columns)
        self.assertTrue("horizon" in forecast.columns)
        self.assertEqual(sorted(list(forecast.horizon.unique())), [0.25, 24.0])

        # check if forecast is indeed of the entire range of the input data
        test_fraction = 0.15
        nb_test = int(np.round(test_fraction * len(self.train_input)))
        validated_data = validation.drop_target_na(
            validation.validate(
                self.pj["id"], self.train_input[-nb_test:], self.pj["flatliner_treshold"]
            )
        )
        data_with_features = TrainFeatureApplicator(
            horizons=[0.25, 24.0], feature_names=self.modelspecs.feature_names
        ).add_features(validated_data, pj=self.pj)
        self.assertEqual(len(forecast), 4 * len(data_with_features))
