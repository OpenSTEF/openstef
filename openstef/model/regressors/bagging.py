"""This module contains the bagging regressor."""

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.utils.validation import check_is_fitted
import pandas as pd


from openstef.model.regressors.regressor import OpenstfRegressor


class BaggingOpenstfRegressor(BaggingRegressor, OpenstfRegressor):
    """Bagging Regressor which implements the Openstf regressor API."""

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs):
        """Fit model."""
        BaggingRegressor.fit(self, x.to_numpy(), y.to_numpy())
        self.feature_names_ = x.columns.tolist()
        self.feature_importances_ = self._get_feature_importance_from_bagging()
        return self

    def _get_feature_importance_from_bagging(self):
        """Aggregate feature importances from all estimators in the BaggingRegressor."""
        check_is_fitted(self, ["feature_names_"])

        # Initialize an array to store cumulative feature importances
        n_features = len(self.feature_names)
        feature_importances = np.zeros(n_features)

        # Aggregate feature importances from each estimator
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            if hasattr(estimator, "feature_importances_"):
                # Only consider the features used by this estimator
                feature_importances[features] += estimator.feature_importances_

        # Normalize the feature importances to sum to 1
        feature_importances /= feature_importances.sum()

        return feature_importances

    @property
    def feature_names(self):
        """The names of he features used to train the model."""
        return self.feature_names_

    @property
    def can_predict_quantiles(self):
        """Indicates wether this model can make quantile predictions."""
        return False
