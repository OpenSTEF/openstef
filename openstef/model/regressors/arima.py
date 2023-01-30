# SPDX-FileCopyrightText: 2017-2022 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains the SARIMAX regressor wrapper around statsmodels implementation."""
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

from openstef.model.regressors.regressor import OpenstfRegressor


class ARIMAOpenstfRegressor(OpenstfRegressor):
    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, x, y, **kwargs):
        # self.ohe_ = OneHotEncoder()
        # exog = self.ohe_.fit_transform(x)
        exog = x
        dates = x.index
        self.model_ = sm.tsa.arima.ARIMA(
            y,
            exog,
            dates=dates,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )
        self.results_ = self.model_.fit()
        self.feature_in_names_ = list(x.columns)
        return self

    def update_historic(self, y_past, x_past):
        # exog_past = self.ohe_(x_past)
        exog_past = x_past
        self.results_ = self.results_.apply(y_past, exog_past)

    def predict(self, x, quantile: float = 0.5, **kwargs):
        start = x.iloc[0].name
        end = x.iloc[-1].name
        # exog = self.ohe_.fit(x)
        exog = x
        if quantile == 0.5:
            predictions = self.results_.predict(start, end, exog=exog).to_numpy()
        elif quantile < 0.5:
            predictions = self.results_.get_prediction(start, end, exog=exog).conf_int(
                alpha=quantile
            )["lower_FC"]
        else:
            predictions = self.results_.get_prediction(start, end, exog=exog).conf_int(
                alpha=1 - quantile
            )["upper FC"]
        return predictions

    def set_feature_importance(self):
        importances = pd.DataFrame(
            {"weight": self.results_.params, "gain": self.results_.pvalues}
        )
        return importances

    @property
    def feature_names(self):
        """The names of he features used to train the model."""
        return self.feature_in_names_

    @property
    def can_predict_quantiles(self):
        """Indicates wether this model can make quantile predictions."""
        return False
