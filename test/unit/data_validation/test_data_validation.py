# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from stf.data_validation.data_validation import validate, clean

from test.utils import BaseTestCase, TestData


class TestDataValidation(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.data_train = TestData.load('input_data_train.pickle')
        self.data_predict = TestData.load('input_data.pickle')

    def test_clean(self):

        cleaned_data = clean(self.data_train)

        self.assertEqual(len(cleaned_data), 11526)

    def test_validate(self):

        self.data_predict['load'][0:50] = 10.0
        validated_data = validate(self.data_predict)

        self.assertEqual(len(validated_data[validated_data['load'].isna()]), 26)


