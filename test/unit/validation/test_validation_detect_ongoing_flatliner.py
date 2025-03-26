# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import UTC, datetime, timedelta
from test.unit.utils.base import BaseTestCase

import numpy as np
import pandas as pd

from openstef.validation.validation import detect_ongoing_flatliner


class TestDetectOngoingFlatliners(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        now = datetime(2023, 10, 30, 12, 0, 0, tzinfo=UTC)
        self.three_hour_range = pd.date_range(
            start=now - timedelta(minutes=180), end=now, freq="0.25H"
        )
        self.four_hour_range_predict_setting = pd.date_range(
            start=now - timedelta(minutes=180),
            end=now + timedelta(minutes=60),
            freq="0.25H",
        )
        self.flatline_value = 3.14

    def test_all_flatline_value(self):
        for detection in [True, False]:
            # Arrange
            load = pd.Series(
                index=self.three_hour_range, data=[self.flatline_value] * 13
            )
            duration_threshold = 120

            # Act
            flatliner_ongoing = detect_ongoing_flatliner(
                load, duration_threshold, detect_non_zero_flatliner=detection
            )

            # Assert
            assert flatliner_ongoing == detection

    def test_all_different(self):
        # Arrange
        load = pd.Series(index=self.three_hour_range, data=[i for i in range(1, 14)])
        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == False

    def test_only_last_different(self):
        # Scenario: A flatliner pattern has been going on for a long time. However,
        # now the pattern has ended since the last measurement is different.

        # Arrange
        load = pd.Series(
            index=self.three_hour_range,
            data=[self.flatline_value for i in range(1, 13)] + [1],
        )
        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == False

    def test_flatliner_pattern_below_threshold(self):
        # Scenario: A flatliner pattern has just begun, however it has not lasted
        # long enough to be considered a flatliner yet (given the configured threshold).

        # Arrange
        load = pd.Series(
            index=self.three_hour_range,
            data=list(range(1, 10)) + [self.flatline_value] * 4,
        )
        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == False

    def test_flatliner_pattern_just_above_threshold(self):
        # Arrange
        load = pd.Series(
            index=self.three_hour_range,
            data=list(range(1, 5)) + [self.flatline_value] * 9,
        )
        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == True

    def test_flatliner_and_missing_values(self):
        # Arrange
        load = pd.Series(
            index=self.three_hour_range,
            data=[1, 2, 3, 4]
            + [self.flatline_value] * 4
            + [np.nan] * 4
            + [self.flatline_value],
        )
        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == True

    def test_all_missing_values(self):
        # Arrange
        load = pd.Series(index=self.three_hour_range, data=[np.nan] * 13)
        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == False

    def test_flatliner_predict_future(self):
        # Scenario: A forecast is made on a flatliner, which contains timestamps in the
        # future with NaN values that need to be predicted.
        # In this case: Time in future > duration_threshold.

        # Arrange
        load = pd.Series(
            index=self.four_hour_range_predict_setting,
            data=[1, 2, 3] + [self.flatline_value] * 9 + [np.nan] * 5,
        )
        duration_threshold = 60

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == True

    def test_flatliner_hovering_around_median_within_rtol(self):
        # Scenario: A flatliner pattern of hovering around a value
        # within the tolerance.

        # Arrange
        load = pd.Series(
            index=self.three_hour_range,
            data=[
                self.flatline_value + (-1) ** i * 1e-6 * self.flatline_value
                for i in range(13)
            ],
        )

        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == True

    def test_flatliner_hovering_around_median_outside_rtol(self):
        # Scenario: A flatliner pattern of hovering around a value
        # outside the tolerance.

        # Arrange
        load = pd.Series(
            index=self.three_hour_range,
            data=[
                self.flatline_value + (-1) ** i * 1e-4 * self.flatline_value
                for i in range(13)
            ],
        )

        duration_threshold = 120

        # Act
        flatliner_ongoing = detect_ongoing_flatliner(
            load, duration_threshold, detect_non_zero_flatliner=True
        )

        # Assert
        assert flatliner_ongoing == False
