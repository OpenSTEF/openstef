"""This module contains the median regressor."""

import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from openstef.model.regressors.regressor import OpenstfRegressor


class MedianRegressor(OpenstfRegressor, RegressorMixin):
    """
    Median regressor implementing the OpenSTEF regressor API.

    This regressor is good for predicting two types of signals:
    - Signals with very slow dynamics compared to the sampling rate, possibly
      with a lot of noise.
    - Signals that switch between two or more states, which random in nature or
    depend on unknown features, but tend to be stable in each state. An example of
    this may be waste heat delivered from an industrial process. Using a median
    over the last few timesteps adds some hysterisis to avoid triggering on noise.
    """

    def __init__(self):
        """Initialize MedianRegressor."""
        super().__init__()

    @property
    def feature_names(self) -> list:
        """Retrieve the model input feature names.

        Returns:
            The list of feature names

        """
        check_is_fitted(self, "feature_names_")
        return self.feature_names_

    @property
    def can_predict_quantiles(self) -> bool:
        return False

    def predict(self, x: pd.DataFrame, **kwargs) -> np.array:
        """
        Predict the median of the lag features for each time step in the context window.

        Args:
            x (pd.DataFrame): The input data for prediction. This should be a pandas dataframe with lag features.

        Returns:
            np.array: The predicted median for each time step in the context window.
            If any lag feature is NaN, this will be ignored.
            If all lag features are NaN, the regressor will return NaN.
        """

        lag_df = x.loc[:, self.feature_names]
        median = lag_df.median(axis=1, skipna=True)

        return median

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> RegressorMixin:
        """This model does not have any hyperparameters to fit,
        but it does need to know the feature names of the lag features.

        Which lag features are used is determined by the feature engineering step.
        """
        self.feature_names_ = list(x.columns[x.columns.str.startswith("T-")])
        if len(self.feature_names_) == 0:
            raise ValueError("No lag features found in the input data.")

        self.feature_importances_ = np.ones(len(self.feature_names_)) / (
            len(self.feature_names_) or 1.0
        )
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True
