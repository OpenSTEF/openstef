# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module defines the DAZL model."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class Dazls(BaseEstimator):
    """DAZLS model.

    The model carries out wind and solar power prediction for unseen target substations using training data from other
    substations with known components.

    """

    model_: Pipeline

    def __init__(self):
        """Initialize DAZL model."""
        self.__name__ = "DAZLS"

        regressor = TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=MinMaxScaler(clip=True),
        )

        self.model_ = Pipeline(
            [("scaler", MinMaxScaler(clip=True)), ("regressor", regressor)]
        )

        # The input columns for the domain and adaptation models (with description)
        self.baseline_input_columns = [
            "radiation",  # Weather parameter
            "windspeed_100m",  # Weather parameter
            "total_load",
        ]
        self.target_columns = ["total_wind_part", "total_solar_part"]

    def fit(self, features, target):
        """Fit the model.

        In this function we scale the input of the domain and adaptation models of the DAZLS MODEL. Then we fit the
        two models. We separate the features into domain_model_input, adaptation_model_input and target, and we use them
        for the fitting and the training of the models.

        Args:
            features: inputs for domain and adaptation model (domain_model_input, adaptation_model_input)
            target: the expected output (y_train)

        """
        x, y = (
            features.loc[:, self.baseline_input_columns],
            target.loc[:, self.target_columns],
        )

        self.model_.fit(x, y)

    def predict(self, x: np.array):
        """Make a prediction.

        For the prediction we use the test data x. We use domain_model_input_columns and
        adaptation_model_input_columns to separate x in test data for domain model and adaptation model respectively.

        There is an option available to return the domain model and adaptation model predictions separately to more
        easily investigate the effectiveness of the models.

        Args:
            x: domain_model_test_data, adaptation_model_test_data
            return_sub_preds : a flag value indicating to return the predictions of the domain model and adaptation
                               model separately. (Default: False.)

        Returns:
            prediction: The output prediction after both models.

        """
        model_test_data = x.loc[:, self.baseline_input_columns]

        return self.model_.predict(model_test_data)

    def score(self, truth, prediction):
        """Evaluation of the prediction's output.

        Args:
            truth: real values
            prediction: predicted values

        Returns:
            RMSE and R2 scores

        """
        rmse = (mean_squared_error(truth, prediction)) ** 0.5
        r2_score_value = r2_score(truth, prediction)
        return rmse, r2_score_value

    def __str__(self):
        """String method of the DAZLs model, provides a summary of the model for easy inspection.

        Returns:
            Summary represented by a string

        """
        summary_str = (
            f"{self.__name__} model summary:\n\n"
            f"Model: {self.model_} \n"
            f"\tInput columns: {self.baseline_input_columns} \n"
            f"\tScaler: {self.model_['scaler']} \n\n"
            f"\tRegressor: {self.model_['regressor']} \n\n"
        )

        return summary_str
