# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pytest

from openstef.exceptions import InputDataOngoingZeroFlatlinerError
from openstef.validation import validation


class TestDataValidation(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.data_train = TestData.load("input_data_train.csv")
        self.data_predict = TestData.load("input_data.csv")
        self.pj = TestData.get_prediction_job(pid=307)

    def test_drop_target_na_no_data_removed(self):
        # Arrange
        original_length = len(self.data_train)

        # Act
        cleaned_data = validation.drop_target_na(self.data_train)

        # Assert
        self.assertEqual(len(cleaned_data), original_length)

    def test_drop_target_na_lots_of_data_removed(self):
        # Arrange
        temp_data = self.data_train.copy()
        temp_data.iloc[1000:, 0] = np.nan

        # Act
        cleaned_data = validation.drop_target_na(temp_data)

        # Assert
        # Data should be removed, since all rows have NaN load except the first 1000 rows
        self.assertEqual(len(cleaned_data), 1000)

    def test_validate(self):
        # Arrange
        default_flatliner_threshold_minutes = 1440
        flatliner_pattern_length_quarters = int(
            default_flatliner_threshold_minutes / 15 + 1
        )
        self.data_predict["load"][0:flatliner_pattern_length_quarters] = 10.0

        # Act
        validated_data = validation.validate(
            self.pj["id"],
            self.data_predict,
            self.pj["flatliner_threshold_minutes"],
            resolution_minutes=15,
        )

        # Assert
        assert (
            self.pj["flatliner_threshold_minutes"]
            == default_flatliner_threshold_minutes
        ), (
            "Change the 'default_flatliner_threshold_minutes' in the arrange part of this unittest"
            "to the default value given in the prediction job dataclass."
        )
        self.assertEqual(
            flatliner_pattern_length_quarters, sum(validated_data["load"].isna())
        )

    def test_validate_none_threshold(self):
        """return the input if flatliner_threshold_minutes is None"""
        # Arrange
        input_data = self.data_train
        flatliner_threshold_minutes = None

        # Act
        res = validation.validate(
            self.pj["id"],
            input_data,
            flatliner_threshold_minutes,
            resolution_minutes=15,
        )

        # Assert
        self.assertDataframeEqual(res, input_data)

    def test_validate_no_datetime_index(self):
        """Raise a value error if input data has no datetime index."""
        # Arrange
        input_data = self.data_train
        input_data = input_data.reset_index(drop=True)

        # Act & assert
        with pytest.raises(ValueError):
            validation.validate(
                self.pj["id"],
                input_data,
                flatliner_threshold_minutes=None,
                resolution_minutes=15,
            )

    def test_validate_ongoing_zero_flatliner(self):
        # Arrange
        input_data = self.data_train
        input_data.iloc[-80:, 0] = 0
        flatliner_threshold_minutes = 360
        resolution_minutes = 15

        # Act & assert
        with pytest.raises(InputDataOngoingZeroFlatlinerError):
            validation.validate(
                self.pj["id"],
                input_data,
                flatliner_threshold_minutes,
                resolution_minutes,
            )
