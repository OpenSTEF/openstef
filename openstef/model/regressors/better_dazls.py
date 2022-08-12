import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
import pickle


# DAZLS algorithm
class BetterDazls(BaseEstimator):
    """
    The model carries out wind and solar power prediction for unseen target substations using training data from other substations with
    known components.

    Any data-driven model can be plugged and used as the base for the domain and the adaptation model.
    """
    def __init__(self):
        self.__name__ = "DAZLS"
        self.xscaler = None
        self.x2scaler = None
        self.yscaler = None
        self.domain_model = KNeighborsRegressor(n_neighbors=20, weights='uniform')
        self.adaptation_model = KNeighborsRegressor(n_neighbors=20, weights='uniform')

        self.domain_model_input_columns_index = ["radiation", "windspeed_100m", "total_substation", "lat", "lon",
                                                 "solar_on",
                                                 "wind_on", "hour", "minute", "var0", "var1", "var2", "sem0", "sem1"]
        self.adaptation_model_input_columns_index = ["total_substation", "lat", "lon", "solar_on", "wind_on", "hour",
                                                     "minute", "var0", "var1", "var2", "sem0", "sem1"]
        self.target_columns_index = ["total_wind_part", "total_solar_part"]

    def fit(self, features, target):
        """
        In this function we scale the input of the domain and adaptation models of the DAZLS MODEL.
        Then we fit the two models.
        With the help of the index we separate the features into domain_model_input, adaptation_model_input and the
        target, and we use them for the fitting and the training of the models.

        :param features: inputs for domain and adaptation model (domain_model_input, adaptation_model_input)
        :param target: the expected output (y_train)

        """

        x, x2, y = features.loc[:, self.domain_model_input_columns_index], features.loc[:, self.adaptation_model_input_columns_index], target.loc[:, self.target_columns_index]
        domain_model_input, adaptation_model_input, y_train = shuffle(x, x2, y, random_state=999)  # just shuffling

        xscaler = MinMaxScaler(clip=True)
        x2scaler = MinMaxScaler(clip=True)
        yscaler = MinMaxScaler(clip=True)
        x_scaler = xscaler.fit(domain_model_input)
        x2_scaler = x2scaler.fit(adaptation_model_input)
        y_scaler = yscaler.fit(y_train)
        domain_model_input = x_scaler.transform(domain_model_input)
        adaptation_model_input = x2_scaler.transform(adaptation_model_input)
        y_train = y_scaler.transform(y_train)

        self.domain_model.fit(domain_model_input, y_train)
        domain_model_pred = self.domain_model.predict(domain_model_input)
        adaptation_model_input = np.concatenate((adaptation_model_input, domain_model_pred), axis=1)
        self.adaptation_model.fit(adaptation_model_input, y_train)

        self.xscaler = x_scaler
        self.x2scaler = x2_scaler
        self.yscaler = y_scaler

    def predict(self, test_features):
        """
        For the prediction we use the test data. We use the index to separate the test data for both domain and
        adaptation models.

        :param test_features: domain_model_test_data, adaptation_model_test_data
        :return: unscaled_test_prediction. The output prediction after both models.
        """
        domain_model_test_data, adaptation_model_test_data = test_features.loc[:, self.domain_model_input_columns_index], test_features.loc[:, self.adaptation_model_input_columns_index]

        # Rescale the test_features (if required)
        domain_model_test_data_scaled = self.xscaler.transform(domain_model_test_data)
        adaptation_model_test_data_scaled = self.x2scaler.transform(adaptation_model_test_data)

        # Use the scaled_test_features to make domain_model_prediction
        domain_model_test_data_pred = self.domain_model.predict(domain_model_test_data_scaled)

        # Use the domain_model_prediction to make adaptation_model_prediction
        adaptation_model_test_data_pred = self.adaptation_model.predict(
            np.concatenate([adaptation_model_test_data_scaled, domain_model_test_data_pred], axis=1))

        # Rescale adaptation_model_prediction (if required)
        unscaled_test_prediction = self.yscaler.inverse_transform(adaptation_model_test_data_pred)

        return unscaled_test_prediction

    def score(self, truth, prediction):
        """
        Evaluation of the prediction's output.

        :param truth: real values
        :param prediction: predicted values

        :return: RMSE and R2 scores
        """

        rmse = (mean_squared_error(truth, prediction)) ** 0.5
        r2 = r2_score(truth, prediction)
        return rmse, r2

    def save_model(self, file_location: str):
        with open('better_dazls_stored.pkl', 'wb') as model_file:
            pickle.dump(BetterDazls, model_file)
        pass

    def load_model(self, file_location: str):
        with open('better_dazls_stored.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        pass
