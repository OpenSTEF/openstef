# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import pandas as pd
import numpy as np
import pytest

from openstef.validation import validation


class TestDataValidation(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.data_train = TestData.load("input_data_train.csv")
        self.data_predict = TestData.load("input_data.csv")
        self.pj = TestData.get_prediction_job(pid=307)

    def test_drop_target_na(self):
        # No data should be removed
        original_length = len(self.data_train)
        cleaned_data = validation.drop_target_na(self.data_train)
        self.assertEqual(len(cleaned_data), original_length)

        # Data should be removed, since all rows have NaN load except the first 1000 rows
        temp_data = self.data_train.copy()
        temp_data.iloc[1000:, 0] = np.nan
        cleaned_data = validation.drop_target_na(temp_data)
        self.assertEqual(len(cleaned_data), 1000)

    def test_validate(self):

        self.data_predict["load"][0:50] = 10.0
        validated_data = validation.validate(
            self.pj["id"], self.data_predict, self.pj["flatliner_treshold"]
        )
        self.assertEqual(26, len(validated_data[validated_data["load"].isna()]))

    def test_validate_none_threshold(self):
        """return the input if flatliner_threshold is None"""
        input_data = self.data_train
        res = validation.validate(self.pj["id"], input_data, flatliner_threshold=None)
        self.assertDataframeEqual(res, input_data)

    def test_validate_no_datetime_index(self):
        """Raise a value error if input data has no datetime index."""
        input_data = self.data_train
        input_data = input_data.reset_index(drop=True)
        with pytest.raises(ValueError):
            res = validation.validate(
                self.pj["id"], input_data, flatliner_threshold=None
            )
            self.assertIsInstance(res, pd.DataFrame)
