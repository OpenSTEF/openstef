from datetime import datetime, timedelta
from test.unit.utils.base import BaseTestCase
import numpy as np
import pandas as pd

from openstef.validation.validation import detect_ongoing_zero_flatliner

now = datetime.utcnow()
three_hour_range = pd.date_range(
    start=now - timedelta(minutes=180), end=now, freq="0.25H"
)


class TestDetectOngoingZeroFlatliners(BaseTestCase):
    def test_all_zero(self):
        # Arrange
        load = pd.Series(index=three_hour_range, data=[0 for i in range(13)])
        duration_threshold = 2

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == True

    def test_all_nonzero(self):
        # Arrange
        load = pd.Series(index=three_hour_range, data=[i for i in range(1, 14)])
        duration_threshold = 2

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False

    def test_only_last_nonzero(self):
        # Arrange
        load = pd.Series(index=three_hour_range, data=[0 for i in range(1, 13)] + [1])
        duration_threshold = 2

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False

    def test_zero_flatliner_pattern_below_threshold(self):
        # Arrange
        load = pd.Series(
            index=three_hour_range, data=[i for i in range(1, 10)] + [0, 0, 0, 0]
        )
        duration_threshold = 2

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False

    def test_zero_flatliner_pattern_just_above_threshold(self):
        # Arrange
        load = pd.Series(index=three_hour_range, data=[1, 2, 3, 4] + [0 for i in range(9)])
        duration_threshold = 2

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == True

    def test_zero_flatliner_and_missing_values(self):
        # Arrange
        load = pd.Series(index=three_hour_range, data=[1, 2, 3, 4] + [0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, 0])
        duration_threshold = 2

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == True

    def test_all_missing_values(self):
        # Arrange
        load = pd.Series(index=three_hour_range, data=[np.nan for i in range(13)])
        duration_threshold = 2

        # Act
        zero_flatliner_ongoing = detect_ongoing_zero_flatliner(load, duration_threshold)

        # Assert
        assert zero_flatliner_ongoing == False
