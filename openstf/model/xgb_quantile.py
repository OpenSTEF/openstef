# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Tuple
from functools import partial

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBRegressor
import numpy as np

import openstf.metrics.metrics as metrics

DEFAULT_QUANTILES: Tuple[float, ...] = (0.9, 0.5, 0.1)


class XGBQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, quantiles: Tuple[float, ...] = DEFAULT_QUANTILES):
        """Initialize XGBQunatileRegressor

            Model that provides quantile regression with XGBoost.
            For each desired quantile an XGBoost model is trained,
            these can later be used to predict quantiles.

        Args:
            quantiles (tuple): Tuple with desired quantiles, quantile 0.5 is required.
                For example: (0.1, 0.5, 0.9)
        """
        # Check if quantile 0.5 is pressent this is required
        if 0.5 not in quantiles:
            raise ValueError(
                "Cannot train quantile model as 0.5 is not in requested quantiles!"
            )

        self.quantiles = quantiles

    def fit(self, x: np.array, y: np.array) -> RegressorMixin:
        """Fits xgb quantile model

        Args:
            x (np.array): Feature matrix
            y (np.array): Labels

        Returns:
            Fitted XGBQuantile model

        """

        # Check/validate input
        check_X_y(x, y)

        # Convert input data to np.array (most of the time this is allready the case)
        x = np.array(x)
        y = np.array(y)

        # Get fitting parameters
        params_quantile = self.get_params().copy()

        quantile_models = {}

        for quantile in self.quantiles:
            # Define objective callback functions specifically for desired quantile
            params_quantile["obj"] = partial(
                metrics.xgb_quantile_eval, quantile=quantile
            )
            params_quantile["feval"] = partial(
                metrics.xgb_quantile_obj, quantile=quantile
            )

            # Initialize xgb model
            specific_quantile_model = XGBRegressor()

            # Set hyperparameters
            specific_quantile_model.set_params(params=params_quantile)

            # Train model for this specific quantile
            quantile_models[quantile] = specific_quantile_model.fit(x, y)

        # Update state of the estimator
        self.estimators_ = quantile_models
        self.is_fitted_ = True

        return self

    def predict(self, x: np.array, quantile: float = 0.5) -> np.array:
        """Makes a prediction for a desired quantile

        Args:
            x (np.array): Feature matrix
            quantile (float): Quantile for which a prediciton is desired,
            note that only quantile are available for which a model is trained

        Returns:
            (np.array): prediction

        Raises:
            ValueError in case no model is trained for the requested quantile

        """
        # Check if model is trained for this quantile
        if quantile not in self.quantiles:
            raise ValueError("No model trained for requested quantile!")

        # Check/validate input
        check_array(x)
        check_is_fitted(self)

        # Convert input data to np.array (most of the time this is allready the case)
        x = np.array(x)

        return self.estimators_[quantile].predict(x)
