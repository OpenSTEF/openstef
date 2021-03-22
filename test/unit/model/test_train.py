# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from test.utils.data import TestData
from unittest.mock import patch, MagicMock
import unittest
from plotly import graph_objects as go

import numpy as np
import pandas as pd

from openstf.model.general import MLModelType, split_data_train_validation_test
from openstf.model.serializer.creator import ModelSerializerCreator
from openstf.model.serializer.xgboost.xgboost import XGBModelSerializer
from openstf.model import train
from openstf.model.trainer.creator import ModelTrainerCreator
from openstf.model.trainer.xgboost.xgboost import XGBModelTrainer

from test.utils import BaseTestCase

pj = TestData.get_prediction_job(pid=307)

datetime_start = datetime.utcnow() - timedelta(days=90)
datetime_end = datetime.utcnow()

data = pd.DataFrame(index=pd.date_range(datetime_start, datetime_end, freq="15T"))

data_table = TestData.load("input_data_train.pickle").head(8641)
# Fill dataframe with values for next test
data["load"] = np.arange(len(data))


# mock functions
def better_than_old_model_mock(_):
    return False


context_mock = MagicMock()
model_trainer_creator_mock = MagicMock()
model_trainer_mock = MagicMock()
model_trainer_mock.better_than_old_model = better_than_old_model_mock

# mock data
split_input_data = train.split_model_data(
    train=pd.DataFrame({"Horizon": [24]}),
    validation=pd.DataFrame(),
    test=pd.DataFrame(),
)
split_predicted_data = train.split_model_data(
    train=None,
    validation=None,
    test=None,
)


class TestTrain(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = data
        self.confidence_interval_ref = TestData.load("confidence_interval.csv")
        (
            self.training_data_ref,
            self.validation_data_ref,
            self.testing_data_ref,
        ) = split_data_train_validation_test(data_table)

        params = {
            "subsample": 0.9,
            "min_child_weight": 4,
            "max_depth": 15,
            "gamma": 0.2,
            "colsample_bytree": 0.85,
            "silent": 1,
            "objective": "reg:squarederror",
        }
        self.model_trainer_ref = XGBModelTrainer(pj)
        self.model_trainer_ref.hyper_parameters.update(params)
        self.model_trainer_ref.train(self.training_data_ref, self.validation_data_ref)

    @patch(
        "openstf.model.train.ModelTrainerCreator",
        MagicMock(return_value=model_trainer_creator_mock),
    )
    def test_create_model_trainer_retrain_young_models_true(
        self,
    ):
        def create_model_trainer_mock():
            model_trainer = MagicMock()
            model_trainer.old_model_age = train.MAX_AGE_YOUNG_MODEL + 1
            return model_trainer

        model_trainer_creator_mock.create_model_trainer = create_model_trainer_mock

        result = train.create_model_trainer(pj, context_mock, retrain_young_models=True)
        self.assertIsInstance(result, MagicMock)

    @patch(
        "openstf.model.train.ModelTrainerCreator",
        MagicMock(return_value=model_trainer_creator_mock),
    )
    def test_create_model_trainer_retrain_young_models_false(self):
        def create_model_trainer_mock():
            model_trainer = MagicMock()
            model_trainer.old_model_age = train.MAX_AGE_YOUNG_MODEL - 1
            return model_trainer

        model_trainer_creator_mock.create_model_trainer = create_model_trainer_mock

        result = train.create_model_trainer(
            pj, context_mock, retrain_young_models=False
        )
        # context logger gecalled
        self.assertEqual(context_mock.logger.info.call_count, 1)
        self.assertIsNone(result)

    @patch("openstf.model.train.pre_process_data")
    @patch("openstf.model.train.is_data_sufficient")
    @patch(
        "openstf.model.train.split_data_train_validation_test",
        MagicMock(return_value=[None, None, None]),
    )
    def test_preprocess_for_model_training(
        self,
        is_data_sufficient_mock,
        pre_process_data_mock,
    ):
        result = train.preprocess_for_model_training(pj, context_mock)
        result_expected = split_predicted_data

        self.assertEqual(pre_process_data_mock.call_count, 1)
        self.assertEqual(is_data_sufficient_mock.call_count, 1)

        self.assertEqual(result, result_expected)

    @patch("openstf.model.train.PredictionModelCreator")
    def test_predict_after_model_training(
        self,
        prediction_model_creator_mock,
    ):
        result = train.predict_after_model_training(
            pj, model_trainer_mock, split_input_data
        )
        result_expected = split_predicted_data

        self.assertEqual(result._fields, result_expected._fields)

    def test_is_new_model_better_compare_to_old_true(self):
        model_trainer_mock.better_than_old_model = better_than_old_model_mock
        result = train.is_new_model_better(
            pj,
            context_mock,
            model_trainer_mock,
            split_input_data,
            compare_to_old=True,
        )
        self.assertFalse(result)

    def test_is_new_model_better_compare_to_old_false(self):
        model_trainer_mock.better_than_old_model = better_than_old_model_mock
        result = train.is_new_model_better(
            pj,
            context_mock,
            model_trainer_mock,
            split_input_data,
            compare_to_old=False,
        )
        self.assertTrue(result)

    @patch(
        "openstf.model.train.plot_feature_importance",
        MagicMock(return_value=go.Figure()),
    )
    @patch("openstf.model.train.plot_data_series", MagicMock(return_value=go.Figure()))
    def test_create_evaluation_figures(self):
        model_trainer = MagicMock()

        result = train.create_evaluation_figures(
            model_trainer, split_input_data, split_predicted_data
        )
        expected_result = go.Figure(), {"Predictor24": go.Figure()}

        self.assertEqual(result, expected_result)

    @patch("openstf.model.train.os.makedirs")
    @patch(
        "openstf.model.train.create_evaluation_figures",
        MagicMock(return_value=(MagicMock(), MagicMock())),
    )
    def test_write_results_new_model_better_model(self, makedirs_mock):
        train.write_results_new_model(
            pj,
            model_trainer_mock,
            True,
            split_input_data,
            split_predicted_data,
            "path_to_save",
        )
        self.assertEqual(makedirs_mock.call_count, 1)

    @patch("openstf.model.train.os.makedirs")
    @patch(
        "openstf.model.train.create_evaluation_figures",
        MagicMock(return_value=(MagicMock(), MagicMock())),
    )
    def test_write_results_new_model_worse_model(self, makedirs_mock):
        train.write_results_new_model(
            pj,
            model_trainer_mock,
            False,
            split_input_data,
            split_predicted_data,
            "path_to_save",
        )
        self.assertEqual(makedirs_mock.call_count, 1)

    @patch("openstf.model.train.send_report_teams_better")
    def test_send_teams_message(self, send_report_teams_better_mock):

        train.send_report_teams(pj, model_trainer_mock, True)
        self.assertEqual(send_report_teams_better_mock.call_count, 1)

    @patch("openstf.model.train.send_report_teams_worse")
    def test_send_teams_message_worse_model(self, send_report_teams_worse_mock):

        train.send_report_teams(pj, model_trainer_mock, False)
        self.assertEqual(send_report_teams_worse_mock.call_count, 1)

    def test_model_trainer_creator(self):
        serializer_creator = ModelSerializerCreator()
        serializer = serializer_creator.create_model_serializer(MLModelType("xgb"))
        model, model_file = serializer.load(pj["id"], TestData.TRAINED_MODELS_FOLDER)

        model_path = (
            TestData.TRAINED_MODELS_FOLDER / f"{pj['id']}/20191119120000/model.bin"
        )

        with patch.object(XGBModelSerializer, "load", return_value=(model, model_path)):
            model_trainer_creator = ModelTrainerCreator(pj)

            model_trainer = model_trainer_creator.create_model_trainer()

            self.assertEqual(type(model_trainer), XGBModelTrainer)

    def test_split_data_train_validation_test(self):
        train_data, validation_data, test_data = split_data_train_validation_test(
            self.data, period_sampling=False
        )
        self.assertEqual(len(train_data), 7345)
        self.assertEqual(len(validation_data), 1297)
        self.assertEqual(len(test_data), 1)

    def test_split_data_train_validation_test_period(self):
        train_data, validation_data, test_data = split_data_train_validation_test(
            self.data, period_sampling=True
        )
        self.assertEqual(len(train_data), 7345)
        self.assertEqual(len(validation_data), 1296)
        self.assertEqual(len(test_data), 1)

    def test_train_checks(self, data=data):
        # Happy flow
        sufficient = train.is_data_sufficient(data)
        self.assertTrue(sufficient)

        # Make 20% of data np.nan to simulate incompleteness that is still acceptable
        data.iloc[0 : int(np.round(0.2 * len(data))), :] = np.nan
        sufficient = train.is_data_sufficient(data)
        self.assertTrue(sufficient)

        # Only pas first 50 rows
        sufficient = train.is_data_sufficient(data.iloc[0:50, :])
        self.assertFalse(sufficient)

        # Make 60% of data np.nan to simulate incompleteness that is not acceptable
        data.iloc[0 : int(np.round(0.6 * len(data))), :] = np.nan
        sufficient = train.is_data_sufficient(data)
        self.assertFalse(sufficient)

    def test_xgboost_model_trainer_train_and_confidence_interval(self):
        # Train
        model = self.model_trainer_ref.train(
            self.training_data_ref, self.validation_data_ref
        )

        # verify a model has been trained
        self.assertTrue(hasattr(model, "predict"))

        # Calculate confidence interval
        confidence_interval = self.model_trainer_ref.calculate_confidence_interval(
            self.testing_data_ref
        )

        # Check if same as reference
        pd.testing.assert_index_equal(
            self.confidence_interval_ref.index, confidence_interval.index
        )


# Run all tests
if __name__ == "__main__":
    unittest.main()
