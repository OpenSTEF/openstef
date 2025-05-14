# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import numpy as np
import pandas as pd
import sklearn

from openstef.model.regressors.bagging import BaggingOpenstfRegressor


class TestBaggingOpenstfRegressor(BaseTestCase):
    def setUp(self):
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    def test_fit(self):
        """Test happy flow of the training of model"""
        model = BaggingOpenstfRegressor()
        model.fit(self.train_input.iloc[:, 1:], self.train_input.iloc[:, 0])

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertIsInstance(model, sklearn.base.BaseEstimator)

    def test_bagging_handles_some_missing_data_in_features(self):
        """Test that the model can handle some missing data"""
        model = BaggingOpenstfRegressor()
        training_data = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "x1": [np.nan, 5, 6],
                "x2": [3, 8, np.nan],
            }
        )

        # Act
        model.fit(training_data[["x1", "x2"]], training_data[["y"]])
        predictions = model.predict(training_data[["x1", "x2"]])

        # Assert
        # Check if the predictions are not NaN
        assert not np.any(np.isnan(predictions))
