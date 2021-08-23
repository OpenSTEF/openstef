# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
import pandas as pd
import structlog
from typing import Union
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class AbstractMLModel(BaseEstimator, RegressorMixin, ABC):
    """
    This class defines an abstract ML model,
    of which concrete implementation can be used as a workhorse throughout openstf

    If you want to include a new ML model in openstf, it should adhere to the interface
    defined here
    """

    def __init__(self) -> None:
        self.logger = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.array:
        pass

    @abstractmethod
    def fit(self, x: np.array, y: np.array, **kwargs) -> RegressorMixin:
        pass

    def score(self, forecast: np.array, actual: np.array) -> float:
        """By default, score returns the R^2 score."""
        return RegressorMixin.score(forecast, actual)

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """Method to set hyperparams of the model, which are used in 'fit'"""
        pass

    ##### Currently, feature importance is calculated quite xgb/lgb specific.
    # It requires that:
    # - model.importance_type can be set to 'weight' or 'gain'
    # - model.feature_importances_ exists (after training) which returns a np.array of feature importances
    # - model._Booster.feature_names returns a list of feature names
    @property
    @abstractmethod
    def importance_type(self):
        pass

    @importance_type.setter
    @abstractmethod
    def importance_type(self, importance_type: str):
        if importance_type not in ["gain", "weight"]:
            raise NotImplementedError("importance_type should be in [gain, weight]")

    @property
    @abstractmethod
    def feature_importances_(self):
        """Note that this method should depend on self.importance_type"""
        pass

    @abstractmethod
    def _Booster(self):
        """model._Booster.feature_names is expected.
        TODO Add requirement for feature_names
        """
        pass

    #### Properties/methods which are currently not part of the model definition,
    # but *are* model specific, which are currently implemented elsewhere in the codebase

    # @property
    # def valid_model_kwargs(self) -> List:
    #     """Should be a list of valid kwargs for the ml model
    #     if not implemented, returns None.
    #     (therefore this is not an abstractmethod)"""
    #     return

    # @abstractmethod
    # def objective(self):
    #     """Objective used for fit / hyperparam optimization"""
    #     pass
