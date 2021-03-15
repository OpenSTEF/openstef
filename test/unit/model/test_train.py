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
from openstf.model.train import (
    is_data_sufficient,
    create_model_for_specific_pj,
    preprocess_for_specific_pj,
    predict_after_training_for_specific_pj,
    create_evaluation_figures_for_specific_pj,
    evaluate_new_model_for_specific_pj,
    MAX_AGE_YOUNG_MODEL,
)
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

params = {
    "subsample": 0.9,
    "min_child_weight": 4,
    "max_depth": 15,
    "gamma": 0.2,
    "colsample_bytree": 0.85,
    "silent": 1,
    "objective": "reg:linear",
}

pj_mock = MagicMock()
context_mock = MagicMock()
model_trainer_mock = MagicMock()

split_input_data = {
    "train_data": pd.DataFrame({"Horizon": [24]}),
    "validation_data": pd.DataFrame(),
    "test_data": pd.DataFrame(),
}
split_predicted_data = {
    "train_predict": None,
    "validation_predict": None,
    "test_predict": None,
}


# def model_trainer_mock():
#     model_trainer = MagicMock()
#     model_trainer.old_model_age = MagicMock(return_value=6)
#     return model_trainer


def model_trainer_creator_mock(pj):
    model_trainer_creator = MagicMock()
    model_trainer_creator.create_model_trainer = model_trainer_mock()
    return model_trainer_creator


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
            "training_period_days": 7,
            "featureset_name": "D",
        }
        self.model_trainer_ref = XGBModelTrainer(pj)
        self.model_trainer_ref.hyper_parameters.update(params)
        self.model_trainer_ref.train(self.training_data_ref, self.validation_data_ref)

    # def test_create_model_for_specific_pj_retrain_young_models_true(
    #     self,
    # ):
    #     context = MagicMock()

    #     result = create_model_for_specific_pj(pj, context, retrain_young_models=True)
    #     self.assertEqual(model_trainer_mock.call_count, 1)
    #     self.assertEqual(result, model_trainer_mock)

    # # @patch(
    # #     "openstf.model.train.model_trainer.old_model_age",
    # #     MagicMock(return_value=MAX_AGE_YOUNG_MODEL + 1),
    # # )
    # def test_create_model_for_specific_pj_retrain_young_models_false(self):
    #     context = MagicMock()
    #     result = create_model_for_specific_pj(pj, context, retrain_young_models=False)
    #     # context logger gecalled
    #     self.assertEqual(context.logger.info.call_count, 1)
    #     self.assertEqual(model_trainer_mock.old_model_age.call_count, 1)
    #     self.assertIsNone(result)

    @patch("openstf.model.train.pre_process_data")
    @patch("openstf.model.train.is_data_sufficient")
    @patch(
        "openstf.model.train.split_data_train_validation_test",
        MagicMock(return_value=[None, None, None]),
    )
    def test_preprocess_for_specific_pj(
        self,
        is_data_sufficient_mock,
        pre_process_data_mock,
    ):
        result = preprocess_for_specific_pj(pj_mock, context_mock)
        result_expected = {
            "train_data": None,
            "validation_data": None,
            "test_data": None,
        }

        self.assertEqual(pre_process_data_mock.call_count, 1)
        self.assertEqual(is_data_sufficient_mock.call_count, 1)

        self.assertEqual(result, result_expected)

    @patch("openstf.model.train.PredictionModelCreator")
    def test_predict_after_training_for_specific_pj(
        self,
        prediction_model_creator_mock,
    ):
        result = predict_after_training_for_specific_pj(
            pj_mock, model_trainer_mock, split_input_data
        )
        # return dict with three keys
        result_expected = split_predicted_data

        self.assertEqual(result.keys(), result_expected.keys())

    @patch(
        "openstf.model.train.plot_feature_importance",
        MagicMock(return_value=go.Figure()),
    )
    @patch("openstf.model.train.plot_data_series", MagicMock(return_value=go.Figure()))
    def test_create_evaluation_figures_for_specific_pj(self):
        model_trainer = MagicMock()

        result = create_evaluation_figures_for_specific_pj(
            model_trainer, split_input_data, split_predicted_data
        )
        expected_result = go.Figure(), {"Predictor24": go.Figure()}

        self.assertEqual(result, expected_result)

    # @patch(
    #     "openstf.model.train.model_trainer.better_than_old_model",
    #     MagicMock(return_value=True),
    # )

    @patch("openstf.model.train.send_report_teams_better")
    @patch("openstf.model.train.os.makedirs")
    @patch(
        "openstf.model.train.create_evaluation_figures_for_specific_pj",
        MagicMock(return_value=(go.Figure(), go.Figure())),
    )
    def test_evaluate_new_model_for_specific_pj_compare_to_old_true(
        self,
        makedirs_mock,
        send_report_teams_better_mock,
    ):

        evaluate_new_model_for_specific_pj(
            pj_mock,
            context_mock,
            model_trainer_mock,
            split_input_data,
            split_predicted_data,
            compare_to_old=True,
        )

        self.assertEqual(makedirs_mock.call_count, 1)
        self.assertEqual(send_report_teams_better_mock.call_count, 1)

    # @patch(
    #     "openstf.model.train.model_trainer.better_than_old_model",
    #     MagicMock(return_value=False),
    # )
    @patch("openstf.model.train.send_report_teams_worse")
    @patch(
        "openstf.model.train.create_evaluation_figures_for_specific_pj",
        MagicMock(return_value=(go.Figure(), go.Figure())),
    )
    @patch("openstf.model.train.os.makedirs")
    def test_evaluate_new_model_for_specific_pj_compare_to_old_false(
        self,
        makedirs_mock,
        send_report_teams_worse_mock,
    ):

        evaluate_new_model_for_specific_pj(
            pj_mock,
            context_mock,
            model_trainer_mock,
            split_input_data,
            split_predicted_data,
            compare_to_old=False,
        )

        self.assertEqual(makedirs_mock.call_count, 1)
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
        sufficient = is_data_sufficient(data)
        self.assertTrue(sufficient)

        # Make 20% of data np.nan to simulate incompleteness that is still acceptable
        data.iloc[0 : int(np.round(0.2 * len(data))), :] = np.nan
        sufficient = is_data_sufficient(data)
        self.assertTrue(sufficient)

        # Only pas first 50 rows
        sufficient = is_data_sufficient(data.iloc[0:50, :])
        self.assertFalse(sufficient)

        # Make 60% of data np.nan to simulate incompleteness that is not acceptable
        data.iloc[0 : int(np.round(0.6 * len(data))), :] = np.nan
        sufficient = is_data_sufficient(data)
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
