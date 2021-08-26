# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
import pandas as pd
import structlog
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class AbstractStfRegressor(BaseEstimator, RegressorMixin, ABC):
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


# In the future, expand this class with:
# - calculate_feature_importance()
# - optimize_hyperparameters()
