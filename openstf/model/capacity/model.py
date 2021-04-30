# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb


class CapacityPredictionModel:
    def __init__(self, classes=None, hyper_params=None):
        """set default hyper-parameters"""
        if hyper_params is None:
            self.hyper_params = {
                "objective": "multi:softprob",
                "learning_rate": 0.01,
                "col_sample_bytree": 0.85,
                "max_depth": 3,
                "n_estimators": 256,
                "verbosity": 0,
            }

        # definition of classification classes
        self.classes = classes

        # create xgb model
        self.model = xgb.XGBClassifier(kwargs=self.hyper_params)

    def train(self, train_x, train_y, val_x, val_y):
        """train model"""
        self.model.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (val_x, val_y)],
            verbose=False,
        )

    def predict(self, x):
        # find best iteration (on validation set)
        best_iter = int(
            np.argmin(self.model.evals_result()["validation_1"]["mlogloss"])
        )

        # predict classes
        y_pred = self.model.predict(x, ntree_limit=best_iter)
        y_pred = pd.DataFrame(y_pred.flatten(), index=x.index)[0]

        # predict probabilities
        y_pred_prob = self.model.predict_proba(x)
        y_pred_prob = pd.DataFrame(y_pred_prob, index=x.index)

        return y_pred, y_pred_prob

    def evaluate(self, x, y_true):
        scores = {}

        # predict on x
        y_pred, _ = self.predict(x)

        # compute f1 score
        scores["f1"] = sklearn.metrics.f1_score(
            y_true.values,
            y_pred.values.flatten(),
            average="weighted",
            labels=np.unique(y_pred.values.flatten()),
        )

        # compute accuracy score
        scores["accuracy"] = sklearn.metrics.accuracy_score(
            y_true.values, y_pred.values.flatten()
        )

        return scores

    def save(self, directory):
        # check directory
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # save model
        with open(directory / "model.pkl", "wb") as fh:
            pickle.dump(self.model, fh)

        # save classes
        with open(directory / "classes.pkl", "wb") as fh:
            pickle.dump(self.classes, fh)

    def load(self, directory):
        # load model
        model_file = Path(directory) / "model.pkl"
        with open(model_file, "rb") as fh:
            self.model = pickle.load(fh)

        # load classes
        classes_file = Path(directory) / "classes.pkl"
        with open(classes_file, "rb") as fh:
            self.classes = pickle.load(fh)
