# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from openstf.validation import validation

from unittest.mock import patch, MagicMock


from test.utils import BaseTestCase, TestData


class TestDataValidation(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.data_train = TestData.load("input_data_train.pickle")
        self.data_predict = TestData.load("input_data.pickle")
        self.pj = TestData.get_prediction_job(pid=307)

    def test_clean(self):

        cleaned_data = validation.clean(self.data_train)

        self.assertEqual(len(cleaned_data), 11526)

    def test_validate(self):

        self.data_predict["load"][0:50] = 10.0
        validated_data = validation.validate(self.pj["id"], self.data_predict)
        self.assertEqual(len(validated_data[validated_data["load"].isna()]), 26)

    @patch("openstf.validation.validation.find_nonzero_flatliner")
    @patch("openstf.validation.validation.replace_invalid_data")
    def test_pre_process_input_data(
        self, replace_invalid_data_mock, nonzero_flatliner_mock
    ):
        suspicious_moments = True

        null_row = MagicMock()
        null_row.isnull.return_value = [True]
        processed_input_data_rows = [(0, null_row), (1, null_row)]
        processed_input_data = MagicMock()
        processed_input_data.iterrows.return_value = processed_input_data_rows

        nonzero_flatliner_mock.return_value = suspicious_moments
        replace_invalid_data_mock.return_value = processed_input_data

        validation.validate(self.pj["id"], data=processed_input_data)

        # simply check if all mocks are called
        for mock_func in [nonzero_flatliner_mock, replace_invalid_data_mock]:
            self.assertEqual(mock_func.call_count, 1)
