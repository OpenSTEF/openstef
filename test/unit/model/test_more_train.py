# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import copy
from datetime import datetime, timedelta
from test.utils.data import TestData
from unittest.mock import MagicMock, patch

import numpy as np

from stf.model.general import MLModelType, split_data_train_validation_test
from stf.model.serializer.creator import ModelSerializerCreator
from stf.model.serializer.xgboost.xgboost import XGBModelSerializer
from stf.model.trainer.xgboost.xgboost import XGBModelTrainer

from test.utils import BaseTestCase

serializer_creator = ModelSerializerCreator()
serializer = serializer_creator.create_model_serializer(MLModelType('xgb'))

pj = TestData.get_prediction_job(pid=307)
data_table = TestData.load("input_data_train.pickle").head(8641)

datetime_start = datetime.utcnow() - timedelta(days=90)
datetime_end = datetime.utcnow()

params = {
    "subsample": 0.9,
    "min_child_weight": 4,
    "max_depth": 15,
    "gamma": 0.2,
    "colsample_bytree": 0.85,
    "silent": 1,
    "objective": "reg:linear",
}

# Generate reference training input data
(
    training_data_ref,
    validation_data_ref,
    testing_data_ref,
) = split_data_train_validation_test(data_table)

# When True: generate a new reference model_trainer pickle:
if False:
    model_trainer_ref = XGBModelTrainer(pj)
    model_trainer_ref.hyper_parameters.update(params)
    model_trainer_ref.train(training_data_ref, validation_data_ref)
    TestData.save(model_trainer_ref, "model_trainer_reference.pickle")
else:
    # Get reference model_trainer object
    model_trainer_ref = TestData.load("model_trainer_reference.pickle")


class TestMoreTrain(BaseTestCase):

    def test_better_than_old_model(self):
        model, model_file = serializer.load(123, TestData.TRAINED_MODELS_FOLDER)

        model_path = TestData.TRAINED_MODELS_FOLDER / "<pid>/20191119120000/model.bin"

        with patch.object(XGBModelSerializer, 'load', return_value=(model, model_path)):
            # Make modeltrainer object
            model_trainer_mock = XGBModelTrainer(pj)

            # New model not available
            model_trainer_mock.old_model.predict = MagicMock(
                name="predict", return_value=testing_data_ref["load"] * 2
            )
            # Untrained model does not make better predictions than old model
            result = model_trainer_mock.better_than_old_model(testing_data_ref)
            self.assertFalse(result)

            # New (trained) model makes better prediction than old model
            model_trainer_mock = copy.deepcopy(model_trainer_ref)
            result = model_trainer_mock.better_than_old_model(testing_data_ref)
            self.assertTrue(result)

            # Old model makes better prediction than new model
            model_trainer_mock.old_model.predict = MagicMock(
                name="predict", return_value=testing_data_ref["load"] * 1.2
            )
            model_trainer_mock.trained_model.predict = MagicMock(
                name="predict", return_value=testing_data_ref["load"] * 2
            )
            result = model_trainer_mock.better_than_old_model(testing_data_ref)
            self.assertFalse(result)

            # Old model not available
            model_trainer_mock.old_model.predict = MagicMock(
                name="predict", return_value=np.nan
            )
            model_trainer_mock.trained_model.predict = MagicMock(
                name="predict", return_value=testing_data_ref["load"] * 2
            )
            result = model_trainer_mock.better_than_old_model(testing_data_ref)
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
