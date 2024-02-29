# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module defines the DAZL model."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


class Dazls(BaseEstimator):
    """DAZLS model.

    The model carries out wind and solar power prediction for unseen target substations using training data from
    other substations with known components.
    This model has two sub-models:

    - domain model : a model taking a set of 'input' features of a substation and make an 'initial' prediction.
      Input features can be features such as: weather, geospatial, total load, etc.
      These features are always directly related to the components' size in some way.


    - adaptation model : a model taking a set of 'meta' features of a substation and refines the domain model's
      prediction. Next to the features, it is trained on the domain model's predictions.
      'Meta' features are features related to the uncertainty of the data, and include:
      variance of the total load, standard deviation of the total load, etc.

    Any data-driven model can be plugged and used as the base for the domain and the adaptation model.

    CAUTION : 'Meta' features should be kept out of the domain model, and vice versa input features should be
    kept out the adaptation model.

    For a full reference, see:
    Teng, S.Y., van Nooten, C. C., van Doorn, J.M., Ottenbros, A., Huijbregts, M., Jansen, J.J.
    Improving Near Real-Time Predictions of Renewable Electricity Production at Substation Level (Submitted)

    Args:
        - BaseEstimator (object) : a base model that can be used to carry out predictions.

    """

    def __init__(self):
        """Initialize DAZL model."""
        self.__name__ = "DAZLS"
        self.domain_model_scaler = MinMaxScaler(clip=True)
        self.adaptation_model_scaler = MinMaxScaler(clip=True)
        self.target_scaler = MinMaxScaler(clip=True)
        self.domain_model = KNeighborsRegressor(n_neighbors=20, weights="uniform")
        self.adaptation_model = KNeighborsRegressor(n_neighbors=20, weights="uniform")

        # The input columns for the domain and adaptation models (with description)
        self.domain_model_input_columns = [
            "radiation",  # Weather parameter
            "windspeed_100m",  # Weather parameter
            "total_substation",  # Substation's measured total load
            "lat",  # Latitude
            "lon",  # Longitude
            "hour",  # Hour of the day
            "minute",  # Minute of the hour
        ]
        self.adaptation_model_input_columns = [
            "var_total",  # Variance of the total load
            "sem_total",  # Standard Error of the Mean of the total load
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
        x, x2, y = (
            features.loc[:, self.domain_model_input_columns],
            features.loc[:, self.adaptation_model_input_columns],
            target.loc[:, self.target_columns],
        )
        domain_model_input, adaptation_model_input, y_train = shuffle(
            x, x2, y, random_state=999
        )  # just shuffling

        self.domain_model_scaler.fit(domain_model_input)
        self.adaptation_model_scaler.fit(adaptation_model_input)
        self.target_scaler.fit(y_train)
        domain_model_input = self.domain_model_scaler.transform(domain_model_input)
        adaptation_model_input = self.adaptation_model_scaler.transform(
            adaptation_model_input
        )
        y_train = self.target_scaler.transform(y_train)

        self.domain_model.fit(domain_model_input, y_train)
        domain_model_pred = self.domain_model.predict(domain_model_input)
        adaptation_model_input = np.concatenate(
            (adaptation_model_input, domain_model_pred), axis=1
        )
        self.adaptation_model.fit(adaptation_model_input, y_train)

    def predict(self, x: np.array, return_sub_preds: bool = False):
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
        domain_model_test_data, adaptation_model_test_data = (
            x.loc[:, self.domain_model_input_columns],
            x.loc[:, self.adaptation_model_input_columns],
        )
        # Rescale test data for both models (if required)
        domain_model_test_data_scaled = self.domain_model_scaler.transform(
            domain_model_test_data
        )
        adaptation_model_test_data_scaled = self.adaptation_model_scaler.transform(
            adaptation_model_test_data
        )
        # Use the scaled data to make domain_model_prediction
        domain_model_test_data_pred = self.domain_model.predict(
            domain_model_test_data_scaled
        )
        # Use the domain_model_prediction to make adaptation_model_prediction
        adaptation_model_test_data_pred = self.adaptation_model.predict(
            np.concatenate(
                [adaptation_model_test_data_scaled, domain_model_test_data_pred], axis=1
            )
        )
        # Rescale adaptation_model_prediction (if required)
        prediction = self.target_scaler.inverse_transform(
            adaptation_model_test_data_pred
        )

        if return_sub_preds:
            prediction_domain = self.target_scaler.inverse_transform(
                domain_model_test_data_pred
            )
            return prediction, prediction_domain
        else:
            return prediction

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
            f"Domain Model: {self.domain_model} \n"
            f"\tInput columns: {self.domain_model_input_columns} \n"
            f"\tScaler: {self.domain_model_scaler} \n\n"
            f"Adaptation Model: {self.adaptation_model} \n"
            f"\tInput columns: {self.adaptation_model_input_columns} \n"
            f"\tScaler: {self.adaptation_model_scaler} \n\n"
            f"Target columns: {self.target_columns}"
        )

        return summary_str
