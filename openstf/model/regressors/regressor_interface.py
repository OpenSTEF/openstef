# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
import pandas as pd
import structlog
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class OpenstfRegressorInterface(BaseEstimator, RegressorMixin, ABC):
    """
    This class defines an abstract ML model,
    of which concrete implementation can be used as a workhorse throughout openstf

    If you want to include a new ML model in openstf, it should adhere to the interface
    defined here
    """

    @abstractmethod
    def predict(self, x: pd.DataFrame, **kwargs) -> np.array:
        """Makes a prediction. Only available after the model has been trained
        Args:
            x (np.array): Feature matrix
            kwargs: model-specific keywords

        Returns:
            (np.array): prediction
        """
        pass

    @abstractmethod
    def fit(self, x: np.array, y: np.array, **kwargs) -> RegressorMixin:
        """Fits the regressor

        Args:
            x (np.array): Feature matrix
            y (np.array): Labels
            kwargs: model-specific keywords

        Returns:
            Fitted model
        """
        pass


# In the future, expand this class with:
# - calculate_feature_importance()
# - optimize_hyperparameters()
