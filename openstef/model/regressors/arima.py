# SPDX-FileCopyrightText: 2017-2022 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains the SARIMAX regressor wrapper around statsmodels implementation."""
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

from openstef.model.regressors.regressor import OpenstfRegressor


class ARIMAOpenstfRegressor(OpenstfRegressor):
    """Wrapper around statmodels implementation of (S)ARIMA(X) model.

    To make prediction, it needs to update its historic, applying the fitted parameters to new data unrelated to the
    original training data.

    """

    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, x, y, **kwargs):
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
        """Apply the fitted parameters to new data unrelated to the original training data. It's a side-effect.

        Creates a new result object using the current fitted parameters,
        applied to a completely new dataset that is assumed to be unrelated to the model’s original data.
        The new results can then be used for analysis or forecasting.
        It should be used before forecasting, to wedge the historic just before the first forecast timestamp.
        Parameters
        ----------
        y_past : pd.DataFrame
            The endogenous (target) data.
        x_past : pd.DataFrame
            The exogenous (features) data.

        """
        exog_past = x_past
        self.results_ = self.results_.apply(y_past, exog_past)

    def predict_quantile(self, start, end, exog, quantile):
        """Quantile prediction.

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting, i.e.,
            the first forecast is start. Can also be a date string to parse or a datetime type.
            Default is the the zeroth observation.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast is end. Can also be a date string to parse or a datetime type.
            However, if the dates index does not have a fixed frequency,
            end must be an integer index if you want out of sample prediction.
            Default is the last observation in the sample.
        exog : pd.DataFrame
            Exogenous data (features).
        quantile : float
            The quantile for the confidence interval.

        Returns
        -------
        pd.Serie
            The quantile prediction.

        """
        alpha = quantile
        idx = 0
        if quantile > 0.5:
            alpha = 1 - quantile
            idx = 1
        return (
            self.results_.get_prediction(start, end, exog=exog)
            .conf_int(alpha=alpha)
            .iloc[:, idx]
        )

    def predict(self, x, quantile: float = 0.5, **kwargs):
        start = x.iloc[0].name
        end = x.iloc[-1].name
        exog = x
        predictions = self.results_.predict(start, end, exog=exog).to_numpy()
        if quantile != 0.5:
            predictions = self.predict_quantile(start, end, exog, quantile)
        return predictions

    def set_feature_importance(self):
        """Because report needs 'weight' and 'gain' as importance metrics, we set the values to these names:

        - 'weight' is corresponding to the coefficients values
        - 'gain' is corresponding to the pvalue for the nullity test of each coefficient

        """
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
        return True
