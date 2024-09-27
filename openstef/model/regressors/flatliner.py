# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import re
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from openstef.model.regressors.regressor import OpenstfRegressor


class FlatlinerRegressor(OpenstfRegressor, RegressorMixin):
    feature_names_: List[str] = []

    def __init__(self, quantiles=None):
        """Initialize FlatlinerRegressor.

        The model always predicts 0.0, regardless of the input features. The model is meant to be used for flatliner
        locations that still expect a prediction while preserving the prediction interface.

        """
        super().__init__()
        self.quantiles = quantiles

    @property
    def feature_names(self) -> list:
        """The names of the features used to train the model."""
        check_is_fitted(self)
        return self.feature_names_

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }

    @property
    def can_predict_quantiles(self) -> bool:
        """Attribute that indicates if the model predict particular quantiles."""
        return True

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs) -> RegressorMixin:
        """Fits flatliner model.

        Args:
            x: Feature matrix
            y: Labels

        Returns:
            Fitted LinearQuantile model

        """
        self.feature_names_ = list(x.columns)
        self.feature_importances_ = np.ones(len(self.feature_names_)) / (
            len(self.feature_names_) or 1.0
        )

        return self

    def predict(self, x: pd.DataFrame, quantile: float = 0.5, **kwargs) -> np.array:
        """Makes a prediction for a desired quantile.

        Args:
            x: Feature matrix
            quantile: Quantile for which a prediciton is desired,
                note that only quantile are available for which a model is trained,
                and that this is a quantile-model specific keyword

        Returns:
            Prediction

        Raises:
            ValueError in case no model is trained for the requested quantile

        """
        check_is_fitted(self)

        return np.zeros(x.shape[0])

    def _get_feature_importance_from_linear(self, quantile: float = 0.5) -> np.array:
        check_is_fitted(self)
        return np.array([0.0 for _ in self.feature_names_])

    @classmethod
    def _get_param_names(cls):
        return [
            "quantiles",
        ]

    def __sklearn_is_fitted__(self) -> bool:
        return True
