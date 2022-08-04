import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, normalize
from sklearn.linear_model import Lasso, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import KernelPCA, PCA
import random
from sklearn.utils import shuffle
from scipy.linalg import fractional_matrix_power
from sklearn.covariance import LedoitWolf, OAS
from sklearn.base import BaseEstimator

# Seed, path, etc,
random.seed(999)
np.random.seed(999)

path = os.path.dirname(os.path.abspath("prep_data"))
folder = ['\\prep_data\\']
combined_data = []
station_name = []

# Read prepared data
for file_name in glob.glob(path + folder[0] + '*.csv'):
    x = pd.read_csv(file_name, low_memory=False, parse_dates=["datetime"])
    x["datetime"] = pd.to_datetime(x["datetime"])
    x = x.set_index('datetime')
    combined_data.append(x)
    sn = os.path.basename(file_name)
    station_name.append(sn[:len(sn) - 4])


# DAZLS algorithm
class DAZLS(BaseEstimator):
    def __init__(self):
        self.__name__ = "DAZLS"
        self.xscaler = None
        self.x2scaler = None
        self.yscaler = None
        self.domain_model = KNeighborsRegressor(n_neighbors=20,
                                  weights='uniform')  # any model can be specified, this is the domain model
        self.adaptation_model = KNeighborsRegressor(n_neighbors=20,
                                   weights='uniform')  # any model can be specified, this is the adaptation model
        self.mini = None
        self.maxi = None
        self.on_off = None
        self.target_substation_pred_data = None
        self.y_test = None


    def fit(self, features, target):


        """
        :param data:
        :param xindex: prep_data columns used to index the domain_model_input [TRAINING DATA COLUMNS DOMAIN_MOD]
        :param x2index: columns being used for adaptation_model_input [TRAINING DATA COLUMNS ADAPT_MODEL]
        :param yindex: columns being used for the target output [TARGET_COLUMNS]
        :param n: (int) substation_index left out training and used to predict
        :param domain_model_clf: classifier 1
        :param adaptation_model_clf: classifier 2
        :param n_delay: (int) =1
        :param cc: (int) =17
        :return:
        """
        domain_model_input_columns_index = ["radiation", "windspeed_100m", "total_substation", "lat", "lon", "solar_on",
                                            "wind_on", "hour", "minute", "var0", "var1", "var2", "sem0", "sem1"]
        adaptation_model_input_columns_index = ["total_substation", "lat", "lon", "solar_on", "wind_on", "hour",
                                                "minute", "var0", "var1", "var2", "sem0", "sem1"]
        target_columns_index = ["total_wind_part", "total_solar_part"]

        x_index = list(set(np.arange(0, nn)) - set([n]))
        y_index = n
        on_off = np.asarray(combined_data[n].iloc[:, [n_delay * 3 + 4, n_delay * 3 + 5]])

        # ###### GPS DIFFERENCE#######
        # diff_index1 = [n_delay * 3 + 2, n_delay * 3 + 3]  # GPS location
        # diff_index2 = list(np.arange(n_delay * 3 + 8, cc))  # Variance and SEM
        #
        # for nx in range(nn):
        #     for ff in diff_index1:
        #         combined_data[nx].iloc[:, ff] = (combined_data[nx].iloc[:, ff] - combined_data[n].iloc[:, ff])
        #     for fff in diff_index2:
        #         combined_data[nx].iloc[:, fff] = (combined_data[nx].iloc[:, fff] - combined_data[n].iloc[:, fff])

        ####################  CALIBRATION #################################
        #temp_data = [combined_data[ind] for ind in x_index]  # Without the target substation
        #ori_data = np.concatenate(temp_data, axis=0)
        test_data = np.asarray(combined_data[y_index])

        X, X2, y = features.loc[:, domain_model_input_columns_index], features.loc[:, adaptation_model_input_columns_index], target.loc[:, target_columns_index]
        domain_model_input, adaptation_model_input, y_train = shuffle(X, X2, y, random_state=999)  # just shuffling
        #domain_model_test_data, adaptation_model_test_data, y_test = test_data[:, xindex], test_data[:, x2index], test_data[:, yindex]
        xscaler = MinMaxScaler(clip=True)
        x2scaler = MinMaxScaler(clip=True)
        yscaler = MinMaxScaler(clip=True)
        X_scaler = xscaler.fit(domain_model_input)
        X2_scaler = x2scaler.fit(adaptation_model_input)
        y_scaler = yscaler.fit(y_train)
        domain_model_input = X_scaler.transform(domain_model_input)
        #domain_model_test_data = X_scaler.transform(domain_model_test_data)
        adaptation_model_input = X2_scaler.transform(adaptation_model_input)
        #adaptation_model_test_data = X2_scaler.transform(adaptation_model_test_data)
        y_train = y_scaler.transform(y_train)
        #y_test = y_scaler.transform(y_test) * on_off

        ###### MIN MAX CAPACITY ######
        mini = np.asarray(test_data[:, [-4, -3]])[-1]
        maxi = np.asarray(test_data[:, [-2, -1]])[-1]
        mini = y_scaler.transform(mini.reshape(1, -1))[0] * on_off
        maxi = y_scaler.transform(maxi.reshape(1, -1))[0] * on_off
        #####################
        ####################

        self.domain_model.fit(domain_model_input, y_train)
        domain_model_pred = self.domain_model.predict(domain_model_input)
        adaptation_model_input = np.concatenate((adaptation_model_input, domain_model_pred), axis=1)
        self.adaptation_model.fit(adaptation_model_input, y_train)

        self.xscaler = X_scaler
        self.x2scaler = X2_scaler
        self.yscaler = y_scaler
        self.mini = mini
        self.maxi = maxi
        self.on_off = on_off

        #self.adaptation_model_pred = adaptation_model.predict(np.concatenate([adaptation_model_test_data, domain_model.predict(domain_model_test_data)], axis=1)) * on_off
        #self.target_substation_pred_data = (self.adaptation_model_pred - np.min(self.adaptation_model_pred, axis=0)) / (
                    #np.max(self.adaptation_model_pred, axis=0) - np.min(self.adaptation_model_pred, axis=0) + 0.0000000000001) * (maxi - mini) + mini
        #self.y_test = y_test

    def predict(self):
        return self.target_substation_pred_data

    def score(self, verbose=True):
        RMSE = (mean_squared_error(self.y_test, self.target_substation_pred_data)) ** 0.5
        R2 = r2_score(self.y_test, self.target_substation_pred_data)
        if verbose:
            print('RMSE test=', RMSE)
            print('R-sqr=', R2)
        return RMSE, R2


###Create delay
n_delay = 1

# CHOOSE THE DATA, METADATA and TARGET, ETC. BY INDEX
cc = len(combined_data[0].columns) - 4
domain_model_input_columns_index = list(np.arange(0, n_delay * 3)) + list(np.arange(n_delay * 3 + 2, cc))
adaptation_model_input_columns_index = list(np.arange(n_delay * 3 + 2, cc))
target_columns_index = [n_delay * 3, n_delay * 3 + 1]

# PREPARATION
ori_combined_data = combined_data.copy()  # Good procedure to prevent data changing in-place
#clf = KNeighborsRegressor(n_neighbors=20, weights='uniform')  # any model can be specified, this is the domain model
#clf2 = KNeighborsRegressor(n_neighbors=20,
#                           weights='uniform')  # any model can be specified, this is the adaptation model

nn = len(station_name)
for n in range(nn):  # loop through all stations (leave one out)
    print(station_name[n])
    model = DAZLS()  # Initialize DAZLS model
    model.fit(training_data.loc[:,feature_columns], training_data.loc[:,target_columns])  # Fit model
    y = model.predict()  # get predicted y
    model.score()  # print prediction performance