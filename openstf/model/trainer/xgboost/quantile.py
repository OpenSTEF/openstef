# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from functools import partial

import xgboost as xgb
import structlog

from openstf.metrics import metrics
from openstf.model.prediction.xgboost.model.quantile import XGBQuantileModel
from openstf.model.trainer.xgboost.xgboost import XGBModelTrainer

# Available trainings period durations for optimization
# After preprocessing, the data consists of 75 days (of the original 90 days).
# To prevent overfitting on a very short period of time, the test and validation
# sets are always a constant fraction of the total data period (75 days).
# We take an optimized fraction of the remaining data as our train-data.
# Given the current fractions for test and validation, the longest train-block
# is 60 days.


class XGBQuantileModelTrainer(XGBModelTrainer):
    def __init__(self, pj):
        super().__init__(pj)
        self.logger = structlog.get_logger(self.__class__.__name__)

    def train(
        self,
        train_data,
        validation_data,
        callbacks=None,
        early_stopping_rounds=10,
        num_boost_round=500,
    ):
        """

        Returns:
            (dict): with keys 'quantile_PXX' and the trained models as values
        """

        # Convert train and validation sets to Dmatrix format for computational
        #  efficiency. Drop Horizon column
        dtrain = xgb.DMatrix(
            train_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore"),
            label=train_data.iloc[:, 0],
        )  # [:,1:-1] excludes label and 'horizon' column
        dval = xgb.DMatrix(
            validation_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore"),
            label=validation_data.iloc[:, 0],
        )

        # Define data set to be monitored during training, the last(validation)
        #  will be used for early stopping
        watchlist = [(dtrain, "train"), (dval, "validation")]

        # Create result dictionary to house the models for the different quantiles
        self.quantile_models = dict()

        # get the xgb (hyper) parameters
        params = {k: self.hyper_parameters[k] for k in self._xgb_hyper_parameter_keys}

        for quantile in self.pj["quantiles"]:
            self.logger.info(f"Training quantile '{quantile}' model", quantile=quantile)

            # Define objective callback functions specifically for desired quantile
            xgb_quantile_eval_this_quantile = partial(
                metrics.xgb_quantile_eval, quantile=quantile
            )
            xgb_quantile_obj_this_quantile = partial(
                metrics.xgb_quantile_obj, quantile=quantile
            )

            # Train quantile model
            self.quantile_models[quantile] = xgb.train(
                params=params,
                dtrain=dtrain,
                evals=watchlist,
                # Can be large because we are early stopping anyway
                num_boost_round=num_boost_round,
                obj=xgb_quantile_obj_this_quantile,
                feval=xgb_quantile_eval_this_quantile,
                verbose_eval=False,
                early_stopping_rounds=early_stopping_rounds,
                callbacks=callbacks,
            )

        self.trained_model = XGBQuantileModel(self.quantile_models)
        return self.trained_model
