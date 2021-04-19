# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import copy
from test.utils.data import TestData
import unittest
from unittest.mock import MagicMock

from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.model.trainer.xgboost.xgboost import XGBModelTrainer

from test.utils import BaseTestCase

pj = TestData.get_prediction_job(pid=307)
data_table = TestData.load("input_data_train.pickle").head(8641)


class TestMoreTrain(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        (
            self.training_data_ref,
            self.validation_data_ref,
            self.testing_data_ref,
        ) = split_data_train_validation_test(data_table)

        # Train model
        params = {
            "subsample": 0.9,
            "min_child_weight": 4,
            "max_depth": 15,
            "gamma": 0.2,
            "colsample_bytree": 0.85,
            "silent": 1,
            "objective": "reg:squarederror",
        }
        model_trainer_ref = XGBModelTrainer(pj)
        model_trainer_ref.hyper_parameters.update(params)
        model_trainer_ref.train(
            self.training_data_ref,
            self.validation_data_ref,
            early_stopping_rounds=1,
            num_boost_round=3,
        )
        self.model_trainer_ref = model_trainer_ref

    def test_better_than_old_model_no_old_model(self):
        model_trainer = XGBModelTrainer(pj)

        # Old model not available
        model_trainer.old_model = None

        model_trainer.trained_model = MagicMock()
        model_trainer.trained_model.predict.return_value = (
            self.testing_data_ref["load"] * 2
        )
        self.assertTrue(model_trainer.better_than_old_model(self.testing_data_ref))

    def test_better_than_old_model_no_new_model(self):
        model_trainer = XGBModelTrainer(pj)

        model_trainer.old_model = MagicMock()
        model_trainer.old_model.predict.return_value = self.testing_data_ref["load"] * 2
        # New model not available
        model_trainer.trained_model = None

        self.assertFalse(model_trainer.better_than_old_model(self.testing_data_ref))

    def test_better_than_old_model_new_model_better(self):
        # New (trained) model makes better prediction than old model
        model_trainer = copy.deepcopy(self.model_trainer_ref)

        self.assertTrue(model_trainer.better_than_old_model(self.testing_data_ref))

    def test_better_than_old_model_old_model_better(self):
        model_trainer = XGBModelTrainer(pj)

        # Old model makes better prediction than new model
        model_trainer.old_model = MagicMock()
        model_trainer.old_model.predict.return_value = (
            self.testing_data_ref["load"] * 1.2
        )

        model_trainer.trained_model = MagicMock()
        model_trainer.trained_model.predict.return_value = (
            self.testing_data_ref["load"] * 2
        )

        # add feature names
        features = ["testfeat1", "testfeat2"]
        model_trainer.old_model.feature_names = features
        model_trainer.trained_model.feature_names = features

        result = model_trainer.better_than_old_model(self.testing_data_ref)

        self.assertFalse(result)

    def test_better_than_old_model_new_features(self):
        model_trainer = XGBModelTrainer(pj)

        # Old model makes better prediction than new model
        model_trainer.old_model = MagicMock()
        model_trainer.old_model.predict.return_value = (
            self.testing_data_ref["load"] * 1.2
        )

        model_trainer.trained_model = MagicMock()
        model_trainer.trained_model.predict.return_value = (
            self.testing_data_ref["load"] * 2
        )

        # add feature names
        model_trainer.old_model.feature_names.return_value = ["testfeat1", "testfeat2"]

        model_trainer.trained_model.feature_names.return_value = [
            "othertestfeat1",
            "testfeat2",
        ]

        result = model_trainer.better_than_old_model(self.testing_data_ref)

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
