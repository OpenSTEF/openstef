# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from openstf.model.metamodels.missing_values_handler import MissingValueHandler
import numpy as np
from sklearn.linear_model import LinearRegression

from openstf.model.regressors.regressor import OpenstfRegressor


class _LinearRegressor(LinearRegression):
    """Wrapper around sklearn linearRegression,
    that provides features importances as absolute values of the coefficients."""

    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        super().fit(x, y)
        self.feature_importances_ = np.abs(self.coef_)


class LinearRegressor(MissingValueHandler):
    def __init__(self, missing_values=np.nan, imputation_strategy=None, fill_value=0):
        super().__init__(
            _LinearRegressor(),
            missing_values=missing_values,
            imputation_strategy=imputation_strategy,
            fill_value=fill_value,
        )


class LinearOpenstfRegressor(LinearRegressor, OpenstfRegressor):
    gain_importance_name = "total_gain"
    weight_importance_name = "weight"

    def __init__(self, missing_values=np.nan, imputation_strategy="mean", fill_value=0):
        super().__init__(
            missing_values=missing_values,
            imputation_strategy=imputation_strategy,
            fill_value=fill_value,
        )

    def fit(self, x, y, **kwargs):
        return super().fit(x, y)

    @property
    def feature_names(self):
        return self.feature_names_
