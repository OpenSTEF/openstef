# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

# import builtins
import unittest
from datetime import datetime
from unittest.mock import patch

# import project modules
from openstf.model.capacity.predict import predict_capacity_prognosis

from test.utils import BaseTestCase, TestData

# define constants for mocking return values
FUCNTION_ARGS = [
    {"id": 1, "name": "job1", "description": "description for job 1"},
    datetime.utcnow().date(),
    datetime.utcnow().date(),
    list(range(13)),
]
LOAD_DATA = TestData.load("capacity_load.csv")
TDCV_DATA = TestData.load("capacity_tdcv_load_profiles.csv")


@patch("openstf.model.capacity.predict.visualize_predictions")
@patch("openstf.model.capacity.predict.prepare_prediction_data")
@patch("openstf.model.capacity.predict.CapacityPredictionModel")
@patch("openstf.model.capacity.predict.apply_capacity_features")
@patch("openstf.model.capacity.predict.DataBase")
@patch("openstf.model.capacity.predict.plotly")
class TestCapacityPredict(BaseTestCase):
    def test_no_exception(
        self,
        plotly_mock,
        db_mock,
        apply_features_mock,
        model_mock,
        prepare_data_mock,
        visualize_predictions_mock,
    ):
        self.add_mock_return_values(
            plotly_mock,
            db_mock,
            apply_features_mock,
            model_mock,
            prepare_data_mock,
            visualize_predictions_mock,
        )
        # run function
        predict_capacity_prognosis(*FUCNTION_ARGS)

        # check mocks
        mocks_called = [
            db_mock.return_value.get_load_pid,
            db_mock.return_value.get_tdcv_load_profiles,
            apply_features_mock,
            prepare_data_mock,
            model_mock,
            model_mock.return_value.predict,
            visualize_predictions_mock,
            plotly_mock.offline.plot,
        ]
        for mock in mocks_called:
            mock.assert_called_once()

    @staticmethod
    def add_mock_return_values(
        plotly_mock,
        db_mock,
        apply_features_mock,
        model_mock,
        prepare_data_mock,
        visualize_predictions_mock,
    ):
        # set database return values
        db_mock.return_value.get_load_pid.return_value = LOAD_DATA
        db_mock.return_value.get_tdcv_load_profiles.return_value = TDCV_DATA
        # set return values for function which return more then 1 argument
        apply_features_mock.return_value = "feature_data", "_"
        model_mock.return_value.predict.return_value = "y_pred", "y_pred_prob"


if __name__ == "__main__":
    unittest.main()
