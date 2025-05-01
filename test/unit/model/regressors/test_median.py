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
                "T-1": [1, 2, 3],
                "T-2": [4, 5, 6],
                "T-3": [7, 8, 9],
                "unrelated_feautre": [10, 11, 12],
            }
        )
        expected_median = pd.Series([4, 5, 6])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        # Check if the predictions are equal to the expected median
        np.testing.assert_allclose(predictions, expected_median)
        # Check that the feature names were captured correctly
        self.assertEqual(model.feature_names_, ["T-1", "T-2", "T-3"])
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
                "T-1": [1, 2, np.nan],
                "T-2": [np.nan, 5, 6],
                "T-3": [3, 8, np.nan],
            }
        )
        expected_median = pd.Series([2, 5, 6])

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
                "T-1": [2, np.nan, np.nan],
                "T-2": [2, np.nan, np.nan],
                "T-3": [5, np.nan, np.nan],
            }
        )
        expected_median = pd.Series([2, np.nan, np.nan])

        # Act
        model.fit(training_data, training_data)
        predictions = model.predict(training_data)

        # Assert
        np.testing.assert_allclose(predictions, expected_median)

    def test_median_fit_with_no_lag_features_raises(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "unrelated_feature": [1, 2, 3],
            }
        )

        # Act
        with self.assertRaisesRegex(
            ValueError, "No lag features found in the input data."
        ):
            model.fit(training_data, training_data)

    def test_predicting_without_fitting_raises(self):
        # Arrange
        model = MedianRegressor()
        training_data = pd.DataFrame(
            {
                "T-1": [1, 2, 3],
                "T-2": [4, 5, 6],
                "T-3": [7, 8, 9],
            }
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
                "T-1": [1, 2, 3],
                "T-2": [4, 5, 6],
                "T-3": [7, 8, 9],
            }
        )

        # Act
        fitted_model = model.fit(training_data, training_data)

        # Assert
        self.assertIsInstance(fitted_model, MedianRegressor)
        self.assertEqual(fitted_model.feature_names_, ["T-1", "T-2", "T-3"])
