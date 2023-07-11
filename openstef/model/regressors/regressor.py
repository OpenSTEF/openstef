# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class OpenstfRegressor(BaseEstimator):
    """This class defines the interface to which all ML models within OpenSTEF should adhere.

    Required methods are indicated by abstractmethods, for which concrete implementations of ML models should have a
    definition. Common functionality which is required for the automated pipelines in OpenSTEF is defined in this class.

    """

    def __init__(self):
        self.feature_importance_dataframe = None
        self.feature_importances_ = None

    def score(self, X, y):
        """Makes `score` method from RegressorMixin available."""
        return RegressorMixin.score(self, X, y)

    ## Define abstract methods required to be implemented by concrete models
    @property
    @abstractmethod
    def feature_names(self) -> list:
        """Retrieve the model input feature names.

        Returns:
            The list of feature names

        """

    @property
    @abstractmethod
    def can_predict_quantiles(self) -> bool:
        """Attribute that indicates if the model predict particular quantiles.

        e.g. XGBQuantileOpenstfRegressor

        """

    @abstractmethod
    def predict(self, x: pd.DataFrame, **kwargs) -> np.array:
        """Makes a prediction. Only available after the model has been trained.

        Args:
            x: Feature matrix
            kwargs: model-specific keywords

        Returns:
            Prediction

        """

    @abstractmethod
    def fit(self, x: np.array, y: np.array, **kwargs) -> RegressorMixin:
        """Fits the regressor.

        Args:
            x: Feature matrix
            y: Labels
            kwargs: model-specific keywords

        Returns:
            Fitted model

        """

    def set_feature_importance(self) -> Union[pd.DataFrame, None]:
        """Get feature importance.

        Returns:
            DataFrame with feature importance.

        """
        # returns a dict if we can get feature importance else returns None
        importance_names = self._get_importance_names()
        # if the model doesn't support feature importance return None
        if importance_names is None:
            return None

        gain = self._fraction_importance(importance_names["gain_importance_name"])
        weight_importance = self._fraction_importance(
            importance_names["weight_importance_name"]
        )

        feature_importance = pd.DataFrame(
            {"gain": gain, "weight": weight_importance}, index=self.feature_names
        )

        feature_importance.sort_values(by="gain", ascending=False, inplace=True)
        return feature_importance

    def _fraction_importance(self, importance: str) -> np.ndarray:
        self.importance_type = importance
        feature_importance = self.feature_importances_
        feature_importance = feature_importance / sum(feature_importance)
        return feature_importance

    @staticmethod
    def _get_importance_names() -> Union[dict, None]:
        """Get importance names if applicable.

        Returns:
            A dict or None, return None if the model can't get feature importance

        """
        return None
