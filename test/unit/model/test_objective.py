# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.utils.base import BaseTestCase
from unittest.mock import MagicMock, patch

from openstf.enums import MLModelType
from openstf.model.objective import XGBRegressorObjective


@patch("openstf.model.objective.split_data_train_validation_test")
@patch("openstf.model.objective.mae")
@patch("openstf.model.objective.ModelCreator")
class TestXGBRegressorObjective(BaseTestCase):
    def test_call(self, model_creator_mock, mea_mock, split_data_mock):

        train_data_mock = MagicMock()
        train_data_mock.columns = ["load", "a", "b", "horizon"]

        split_data_mock.return_value = train_data_mock, train_data_mock, train_data_mock
        input_data = None

        objective = XGBRegressorObjective(input_data)
        trial_mock = MagicMock()
        objective(trial_mock)

        # the first arg to split_data should be input_data
        self.assertEqual(split_data_mock.call_args[0][0], input_data)

        # create model should be called the the right model type
        self.assertEqual(
            model_creator_mock.create_model.call_args[0][0], MLModelType.XGB
        )


if __name__ == "__main__":
    unittest.main()
