import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler , MinMaxScaler,MaxAbsScaler,RobustScaler, Normalizer, normalize
#from chemsy.predict.methods import *
from sklearn.linear_model import Lasso, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
#from chemsy.prep import *
from sklearn.decomposition import KernelPCA, PCA
import random
from sklearn.utils import shuffle
from scipy.linalg import fractional_matrix_power
from sklearn.covariance import LedoitWolf, OAS
from sklearn.base import BaseEstimator

# DAZLS algorithm
class DAZLS(BaseEstimator):
    def __init__(self):
        self.__name__ = "DAZLS"
        self.xscaler = None
        self.x2scaler = None
        self.yscaler = None
        self.domain_model = None  # 1st model
        self.adaptation_model = None  # 2nd model
        self.mini = None  #
        self.maxi = None
        self.on_off = None  # binary target metadata for which solar or wind facility are present.
        self.target_substation_pred_data = None  # the predicted output data of the target substation
        self.y_test = None

    def fit(self, combined_data, xindex, x2index, yindex, n, domain_model_clf, adaptation_model_clf, n_delay, cc):
        x_index = list(set(np.arange(0, nn)) - set([n]))
        y_index = n
        on_off = np.asarray(combined_data[n].iloc[:, [n_delay * 3 + 4, n_delay * 3 + 5]])

        ###### GPS DIFFERENCE#######
        diff_index1 = [n_delay * 3 + 2, n_delay * 3 + 3]  # GPS location
        diff_index2 = list(np.arange(n_delay * 3 + 8, cc))  # Variance and SEM

        for nx in range(nn):
            for ff in diff_index1:
                combined_data[nx].iloc[:, ff] = (combined_data[nx].iloc[:, ff] - combined_data[n].iloc[:, ff])
            for fff in diff_index2:
                combined_data[nx].iloc[:, fff] = (combined_data[nx].iloc[:, fff] - combined_data[n].iloc[:, fff])

        ####################  CALIBRATION #################################
        temp_data = [combined_data[ind] for ind in x_index]  # Without the target substation
        ori_data = np.concatenate(temp_data, axis=0)
        test_data = np.asarray(combined_data[y_index])
        X, X2, y = ori_data[:, xindex], ori_data[:, x2index], ori_data[:, yindex]
        domain_model_input, adaptation_model_input, y_train = shuffle(X, X2, y, random_state=999)  # just shuffling
        domain_model_test_data, adaptation_model_test_data, y_test = test_data[:, xindex], test_data[:,
                                                                                           x2index], test_data[:,
                                                                                                     yindex]
        xscaler = MinMaxScaler(clip=True)  # min-max renormalization
        x2scaler = MinMaxScaler(clip=True)
        yscaler = MinMaxScaler(clip=True)
        X_scaler = xscaler.fit(domain_model_input)  # min-max renormalization of the input of the domain model
        X2_scaler = x2scaler.fit(adaptation_model_input)  # min-max renormalization of the input of the adaptation model
        y_scaler = yscaler.fit(y_train)  #
        domain_model_input = X_scaler.transform(
            domain_model_input)  # uses the above scaler to transform the train set of the first model (I guess it makes it df)
        domain_model_test_data = X_scaler.transform(
            domain_model_test_data)  # << << to transform the test set of the first model
        adaptation_model_input = X2_scaler.transform(adaptation_model_input)
        adaptation_model_test_data = X2_scaler.transform(adaptation_model_test_data)
        y_train = y_scaler.transform(y_train)
        y_test = y_scaler.transform(y_test) * on_off

        ###### MIN MAX CAPACITY ######
        mini = np.asarray(test_data[:, [-4, -3]])[-1]
        maxi = np.asarray(test_data[:, [-2, -1]])[-1]
        mini = y_scaler.transform(mini.reshape(1, -1))[0] * on_off
        maxi = y_scaler.transform(maxi.reshape(1, -1))[0] * on_off
        #####################
        ####################

        domain_model_input_df = pd.DataFrame(domain_model_input)
        domain_model_input = domain_model_input_df.bfill()
        domain_model_input = domain_model_input.values

        domain_model_clf.fit(domain_model_input,
                             y_train)  # the domain model, maps the input data Xs to the output data ys
        domain_model_pred = domain_model_clf.predict(domain_model_input)  # the predicted output of the domain model
        adaptation_model_input = np.concatenate((adaptation_model_input, domain_model_pred),
                                                axis=1)  # the predicted output of the domain model is-
        # concatenated with the source metadata and act as an input for the adaptation model
        adaptation_model_clf.fit(adaptation_model_input,
                                 y_train)  # the adaptation model uses the above as input X2 and predicts again the output data ys

        self.xscaler = X_scaler
        self.x2scaler = X2_scaler
        self.yscaler = y_scaler
        self.domain_model = domain_model_clf
        self.adaptation_model = adaptation_model_clf
        self.mini = mini  #
        self.maxi = maxi  #
        self.on_off = on_off  # ?? binary target metadata for which solar or wind facility are present (B)

        domain_model_test_data_df = pd.DataFrame(domain_model_test_data)
        domain_model_test_data = domain_model_test_data_df.bfill()
        domain_model_test_data = domain_model_test_data.values


        domain_model_pred_test_data = domain_model_clf.predict(domain_model_test_data)
        concatenated_adapt_model_test_data = np.concatenate([adaptation_model_test_data, domain_model_pred_test_data],
                                                            axis=1)
        self.adaptation_model_pred = adaptation_model_clf.predict(concatenated_adapt_model_test_data) * on_off
        # output prediction of the adaptation model. It maps input-output data in the domain model into metadata domain.

        ## ??Then, it multiplies the adaptation_model_pred with the binary target metadata (*on-off)

        self.target_substation_pred_data = (self.adaptation_model_pred - np.min(self.adaptation_model_pred, axis=0)) / (
                np.max(self.adaptation_model_pred, axis=0) - np.min(self.adaptation_model_pred,
                                                                    axis=0) + 0.0000000000001) * (maxi - mini) + mini
        # Then, the output is min-max renormalized to fit between the max and the min capacity of the facility

        self.y_test = y_test

    def predict(self):
        return self.target_substation_pred_data

    def score(self, verbose=True):
        RMSE = (mean_squared_error(self.y_test, self.target_substation_pred_data)) ** 0.5
        R2 = r2_score(self.y_test, self.target_substation_pred_data)
        if verbose:
            print('RMSE test=', RMSE)
            print('R-sqr=', R2)
        return RMSE, R2

#Seed, path, etc,
random.seed(999)
np.random.seed(999)


path = os.path.dirname(os.path.abspath('combined_data'))
folder=['\\combined_data\\']
combined_data=[]
station_name=[]

#Read prepared data
for file_name in glob.glob(path+folder[0]+'*.csv'):
    x = pd.read_csv(file_name, low_memory=False,parse_dates=["datetime"])
    x["datetime"]=pd.to_datetime(x["datetime"])
    x=x.set_index('datetime')
    combined_data.append(x)
    sn=os.path.basename(file_name)
    station_name.append(sn[:len(sn)-4])

n_delay=1
# CHOOSE THE DATA, METADATA and TARGET, ETC. BY INDEX
cc=len(combined_data[0].columns)-4
xindex=list(np.arange(0,n_delay*3))+list(np.arange(n_delay*3+2,cc))
x2index=list(np.arange(n_delay*3+2,cc))
yindex=[n_delay*3,n_delay*3+1]

#PREPARATION
ori_combined_data=combined_data.copy() #Good procedure to prevent data changing in-place
domain_model_clf=KNeighborsRegressor(n_neighbors=20,weights='uniform') #any model can be specified, this is the domain model
adaptation_model_clf=KNeighborsRegressor(n_neighbors=20,weights='uniform') #any model can be specified, this is the adaptation model

nn=len(station_name)
for n in range(nn): #loop through all stations (leave one out)
    print(station_name[n])
    model=DAZLS() #Initialize DAZLS model
    model.fit(combined_data=ori_combined_data, xindex=xindex,x2index=x2index,yindex=yindex,n=n,domain_model_clf=domain_model_clf,adaptation_model_clf=adaptation_model_clf,n_delay=n_delay,cc=cc) #Fit model
    y=model.predict() #get predicted y
    model.score() #print prediction performance

