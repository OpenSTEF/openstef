# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from functools import partial
from typing import Tuple

import numpy as np
import xgboost as xgb
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import Booster

import openstef.metrics.metrics as metrics
from openstef.model.regressors.regressor import OpenstfRegressor

DEFAULT_QUANTILES: Tuple[float, ...] = (0.9, 0.5, 0.1)


class XGBQuantileOpenstfRegressor(OpenstfRegressor):
    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }

    def __init__(
        self,
        quantiles: Tuple[float, ...] = DEFAULT_QUANTILES,
        gamma: float = 0.37879654,
        colsample_bytree: float = 0.78203051,
        subsample: float = 0.9,
        min_child_weight: int = 4,
        max_depth: int = 4,
    ):
        """Initialize XGBQuantileRegressor

            Model that provides quantile regression with XGBoost.
            For each desired quantile an XGBoost model is trained,
            these can later be used to predict quantiles.

        Args:
            quantiles (tuple): Tuple with desired quantiles, quantile 0.5 is required.
                For example: (0.1, 0.5, 0.9)
        """
        super().__init__()
        # Check if quantile 0.5 is pressent this is required
        if 0.5 not in quantiles:
            raise ValueError(
                "Cannot train quantile model as 0.5 is not in requested quantiles!"
            )

        self.quantiles = quantiles

        # Set attributes for hyper parameters
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree

    def fit(self, x: np.array, y: np.array, **kwargs) -> OpenstfRegressor:
        """Fits xgb quantile model

        Args:
            x (np.array): Feature matrix
            y (np.array): Labels

        Returns:
            Fitted XGBQuantile model

        """

        # TODO: specify these required kwargs in the function definition
        early_stopping_rounds = kwargs.get("early_stopping_rounds", None)
        eval_set = kwargs.get("eval_set", None)

        # Check/validate input
        check_X_y(x, y, force_all_finite="allow-nan")

        # Convert x and y to dmatrix input
        dtrain = xgb.DMatrix(x.copy(deep=True), label=y.copy(deep=True))

        # Define watchlist if eval_set is defined
        if eval_set:
            dval = xgb.DMatrix(
                eval_set[1][0].copy(deep=True),
                label=eval_set[1][1].copy(deep=True),
            )

            # Define data set to be monitored during training, the last(validation)
            #  will be used for early stopping
            watchlist = [(dtrain, "train"), (dval, "validation")]
        else:
            watchlist = ()

        # Get fitting parameters - only those required for xgbooster's
        xgb_regressor_params = {
            key: value
            for key, value in self.get_params().items()
            if key in xgb.XGBRegressor().get_params().keys()
        }

        quantile_models = {}

        for quantile in self.quantiles:
            # Define objective callback functions specifically for desired quantile
            xgb_quantile_eval_this_quantile = partial(
                metrics.xgb_quantile_eval, quantile=quantile
            )
            xgb_quantile_obj_this_quantile = partial(
                metrics.xgb_quantile_obj, quantile=quantile
            )

            # Train quantile model
            quantile_models[quantile] = xgb.train(
                params=xgb_regressor_params,
                dtrain=dtrain,
                evals=watchlist,
                # Can be large because we are early stopping anyway
                num_boost_round=100,
                obj=xgb_quantile_obj_this_quantile,
                feval=xgb_quantile_eval_this_quantile,
                verbose_eval=False,
                early_stopping_rounds=early_stopping_rounds,
            )

        # Set weigths and features from the 0.5 (median) model
        self.feature_importances_ = self.get_feature_importances_from_booster(
            quantile_models[0.5]
        )
        self._Booster = quantile_models[0.5]  # Used for feature names later on
        # Update state of the estimator
        self.estimators_ = quantile_models
        self.is_fitted_ = True

        return self

    def predict(self, x: np.array, quantile: float = 0.5) -> np.array:
        """Makes a prediction for a desired quantile

        Args:
            x (np.array): Feature matrix
            quantile (float): Quantile for which a prediciton is desired,
            note that only quantile are available for which a model is trained,
            and that this is a quantile-model specific keyword

        Returns:
            (np.array): prediction

        Raises:
            ValueError in case no model is trained for the requested quantile

        """
        # Check if model is trained for this quantile
        if quantile not in self.quantiles:
            raise ValueError("No model trained for requested quantile!")

        # Check/validate input
        check_array(x, force_all_finite="allow-nan")
        check_is_fitted(self)

        # Convert array to dmatrix
        dmatrix_input = xgb.DMatrix(x.copy(deep=True))

        return self.estimators_[quantile].predict(
            dmatrix_input, ntree_limit=self.estimators_[quantile].best_ntree_limit
        )

    @classmethod
    def get_feature_importances_from_booster(cls, booster: Booster) -> np.ndarray:
        """Gets feauture importances from a XGB booster.
            This is based on the feature_importance_ property defined in:
            https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py

        Args:
            booster(Booster): Booster object,
            most of the times the median model (quantile=0.5) is preferred

        Returns:
            (np.ndarray) with normalized feature importances

        """

        # Get score
        score = booster.get_score(importance_type="gain")

        # Get feature names from booster
        feature_names = booster.feature_names

        # Get importance
        feature_importance = [score.get(f, 0.0) for f in feature_names]
        # Convert to array
        features_importance_array = np.array(feature_importance, dtype=np.float32)

        total = features_importance_array.sum()  # For normalizing
        if total == 0:
            return features_importance_array
        return features_importance_array / total  # Normalize

    @property
    def feature_names(self):
        return self._Booster.feature_names
