# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase

import numpy as np
import pandas as pd

from openstef.model.regressors.median import MedianRegressor


class TestMedianRegressor(BaseTestCase):

    def test_median_returns_median(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, np.nan, np.nan],
                "T-2min": [4, 1, np.nan],
                "T-3min": [7, 4, 1],
                "unrelated_feature": [10, 11, 12],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )
        expected_median = pd.Series([4, 4, 4])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        # Check if the predictions are equal to the expected median
        np.testing.assert_allclose(predictions, expected_median)
        # Check that the feature names were captured correctly
        self.assertEqual(model.feature_names_, ["T-1min", "T-2min", "T-3min"])
        # Check that the feature importances were set correctly
        self.assertIsNotNone(model.feature_importances_)
        self.assertEqual(model.feature_importances_.shape, (3,))
        np.testing.assert_allclose(
            model.feature_importances_, np.array([1 / 3, 1 / 3, 1 / 3])
        )

    def test_median_handles_some_missing_data(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, np.nan, np.nan],
                "T-2min": [np.nan, 1, np.nan],
                "T-3min": [3, np.nan, 1],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )
        expected_median = pd.Series([2, 1.5, 1.5])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        np.testing.assert_allclose(predictions, expected_median)

    def test_median_handles_missing_data_for_some_horizons(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [np.nan, np.nan, np.nan],
                "T-2min": [np.nan, np.nan, np.nan],
                "T-3min": [5, np.nan, np.nan],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )
        expected_median = pd.Series([5, 5, 5])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        np.testing.assert_allclose(predictions, expected_median)

    def test_median_handles_all_missing_data(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [np.nan, np.nan, np.nan],
                "T-2min": [np.nan, np.nan, np.nan],
                "T-3min": [np.nan, np.nan, np.nan],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )
        expected_median = pd.Series([np.nan, np.nan, np.nan])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        np.testing.assert_allclose(predictions, expected_median)

    def test_median_uses_lag_features_if_available(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, 2, 3],
                "T-2min": [4, 5, 6],
                "T-3min": [7, 8, 9],
            },
            index=pd.date_range("2023-01-01T00:00", periods=3, freq="min"),
        )
        excepted_median = pd.Series([4, 5, 6])
        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)
        # Assert
        np.testing.assert_allclose(predictions, excepted_median)


    def test_median_handles_small_gap(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, np.nan, np.nan, np.nan, np.nan],
                "T-2min": [2, 1, np.nan, np.nan, np.nan],
                "T-3min": [3, 2, 1, np.nan, np.nan],
                "T-4min": [4, 3, 2, 1, np.nan],
                "T-5min": [5, 4, 3, 2, 1],
            },
            index=pd.date_range("2023-01-01", periods=5, freq="min"),
        )
        # Remove the second row to create a small gap
        training_data = training_data[training_data.index != "2023-01-01 00:01:00"]
        expected_median = pd.Series([3, 3, 3, 3])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        np.testing.assert_allclose(predictions, expected_median)

    def test_median_handles_large_gap(self):
        # Arrange
        model = MedianRegressor()
        training_data_1 = pd.DataFrame(
            {
                "T-1min": [1, 2, 3],
                "T-2min": [4, 5, 6],
                "T-3min": [7, 8, 9],
            },
            index=pd.date_range("2023-01-01T00:00", periods=3, freq="min"),
        )
        training_data_2 = pd.DataFrame(
            {
                "T-1min": [10, 11, 12],
                "T-2min": [13, 14, 15],
                "T-3min": [16, 17, 18],
            },
            index=pd.date_range("2023-01-02T00:10", periods=3, freq="min"),
        )
        training_data = pd.concat([training_data_1, training_data_2])
        expected_median = pd.Series([4, 5, 6, 13, 14, 15])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        np.testing.assert_allclose(predictions, expected_median)

    def test_median_fit_with_missing_features_raises(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, 2, 3],
                "T-2min": [4, 5, 6],
                "T-3min": [7, 8, 9],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )
        model.fit(training_data, training_data)

        # Act & Assert
        with self.assertRaisesRegex(
            ValueError,
            "The input data is missing the following lag features: {'T-3min'}",
        ):
            model.predict(training_data[["T-1min", "T-2min"]])

    def test_median_fit_with_no_lag_features_raises(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "unrelated_feature": [1, 2, 3],
            }
        )

        # Act & Assert
        with self.assertRaisesRegex(
            ValueError, "No lag features found in the input data."
        ):
            model.fit(training_data, training_data)

    def test_median_fit_with_inconsistent_lag_features_raises(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, 2, 3],
                "T-5min": [4, 5, 6],
                "T-60min": [7, 8, 9],
                "T-4d": [10, 11, 12],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )

        # Act & Assert
        with self.assertRaisesRegex(
            ValueError,
            "Lag features are not evenly spaced",
        ):
            model.fit(training_data, training_data)

    def test_median_fit_with_inconsistent_frequency_raises(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, 2, 3],
                "T-2min": [4, 5, 6],
                "T-3min": [7, 8, 9],
            },
            index=pd.date_range("2023-01-01", periods=3, freq=pd.Timedelta(hours=1)),
        )

        # Act & Assert
        with self.assertRaisesRegex(
            ValueError,
            "does not match the model frequency.",
        ):
            model.fit(training_data, training_data)


    def test_predicting_without_fitting_raises(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, 2, 3],
                "T-2min": [4, 5, 6],
                "T-3min": [7, 8, 9],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )

        # Act & Assert
        with self.assertRaisesRegex(
            AttributeError,
            "This MedianRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
        ):
            model.predict(training_data)

    def test_median_fit_returns_fitted_model(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1min": [1, 2, 3],
                "T-2min": [4, 5, 6],
                "T-3min": [7, 8, 9],
            },
            index=pd.date_range("2023-01-01", periods=3, freq="min"),
        )

        # Act
        fitted_model = model.fit(training_data, training_data)

        # Assert
        self.assertIsInstance(fitted_model, MedianRegressor)
        self.assertEqual(fitted_model.feature_names_, ["T-1min", "T-2min", "T-3min"])

