# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta, UTC
from test.unit.utils.base import BaseTestCase

import numpy as np
import pandas as pd
from freezegun import freeze_time

from openstef.validation.validation import detect_ongoing_zero_flatliner


@freeze_time("2023-10-30 12:01:02")
class TestDetectOngoingZeroFlatliners(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        now = datetime.now(tz=UTC)
        self.three_hour_range = pd.date_range(
            start=now - timedelta(minutes=180), end=now, freq="0.25H"
        )
        self.four_hour_range_predict_setting = pd.date_range(
            start=now - timedelta(minutes=180),
            end=now + timedelta(minutes=60),
            freq="0.25H",
        )

    def test_all_zero(self):
        # Arrange
        load = pd.Series(index=self.three_hour_range, data=[0 for i in range(13)])
        duration_threshold = 120

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == True

    def test_all_nonzero(self):
        # Arrange
        load = pd.Series(index=self.three_hour_range, data=[i for i in range(1, 14)])
        duration_threshold = 120

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False

    def test_only_last_nonzero(self):
        # Scenario: A zero flatliner pattern has been going on for a long time. However,
        # now the pattern has ended since the last measurement is not zero anymore.

        # Arrange
        load = pd.Series(
            index=self.three_hour_range, data=[0 for i in range(1, 13)] + [1]
        )
        duration_threshold = 120

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False

    def test_zero_flatliner_pattern_below_threshold(self):
        # Scenario: A zero flatliner pattern has just begun, however it has not lasted
        # long enough to be considered a zero flatliner yet (given the configured threshold).

        # Arrange
        load = pd.Series(
            index=self.three_hour_range, data=[i for i in range(1, 10)] + [0, 0, 0, 0]
        )
        duration_threshold = 120

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False

    def test_zero_flatliner_pattern_just_above_threshold(self):
        # Arrange
        load = pd.Series(
            index=self.three_hour_range, data=[1, 2, 3, 4] + [0 for i in range(9)]
        )
        duration_threshold = 120

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == True

    def test_zero_flatliner_and_missing_values(self):
        # Arrange
        load = pd.Series(
            index=self.three_hour_range,
            data=[1, 2, 3, 4] + [0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, 0],
        )
        duration_threshold = 120

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == True

    def test_all_missing_values(self):
        # Arrange
        load = pd.Series(index=self.three_hour_range, data=[np.nan for i in range(13)])
        duration_threshold = 120

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False

    def test_zero_flatliner_predict_future(self):
        # Scenario: A forecast is made on a zero flatliner, which contains timestamps in the
        # future with NaN values that need to be predicted.
        # In this case: Time in future > duration_threshold.

        # Arrange
        load = pd.Series(
            index=self.four_hour_range_predict_setting,
            data=[1, 2, 3]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [np.nan, np.nan, np.nan, np.nan, np.nan],
        )
        duration_threshold = 60

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == True
