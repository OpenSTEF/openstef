# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from openstf.model.metamodels.missing_values_handler import MissingValueHandler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from openstf.model.regressors.regressor import OpenstfRegressor


class LinearRegressor(LinearRegression):
    """Wrapper around sklearn linearRegression,
    that provides features importances as absolute values of the coefficients."""

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        super().fit(X, y)
        self.feature_importances_ = np.abs(self.coef_)


class LinearRTEOpenstfRegressor(OpenstfRegressor):
    gain_importance_name = "total_gain"
    weight_importance_name = "weight"

    def __init__(self, imputation_strategy="mean"):
        super().__init__()
        self.imputation_strategy = imputation_strategy

    def _more_tags(self):
        return {"allow_nan": self.imputation_strategy is not None}

    def fit(self, X, y, **kwargs):
        self.estimator_ = MissingValueHandler(
            LinearRegressor(), imputation_strategy=self.imputation_strategy
        )

        self.estimator_.fit(X, y)
        self.feature_names_ = self.estimator_.feature_names_
        self.n_features_in_ = self.estimator_.n_features_in_
        self.feature_importances_ = self.estimator_.feature_importances_

        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)

    @property
    def feature_names(self):
        return self.feature_names_
