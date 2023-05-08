# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains the SARIMAX regressor wrapper around statsmodels implementation."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from openstef.model.regressors.regressor import OpenstfRegressor


class ARIMAOpenstfRegressor(OpenstfRegressor):
    """Wrapper around statmodels implementation of (S)ARIMA(X) model.

    The fit of an ARIMA statsmodels produces a result object which is used to perform the various computations around forecasting.
    (see https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMAResults.html)

    To make a prediction, it needs to update the result object's historic data,
    ie the past values of the target/endogenous data and the features/exogenous data,
    applying the fitted parameters to these new data unrelated to the original training data.
    This update can be performed by the method `update_historic_data`.

    In the following code, we use interchangeably the statmodels and scikit-learn terminology for the variables:
        - the features 'x' is equivalent to the exogenous data: 'exog' for short.
        - the target 'y' is equivalent to the endogenous data: 'endog' for short.
    More information here https://www.statsmodels.org/stable/endog_exog.html.

    """

    def __init__(
        self,
        backtest_max_horizon=1440,
        order=(0, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
    ):
        self.backtest_max_horizon = backtest_max_horizon
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, x, y, **kwargs):
        dates = x.index
        self.model_ = sm.tsa.arima.ARIMA(
            endog=y,
            exog=x,
            dates=dates,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )
        self.results_ = self.model_.fit()
        self.feature_in_names_ = list(x.columns)
        return self

    def update_historic_data(self, x_past, y_past):
        """Apply the fitted parameters to new data unrelated to the original training data. It's a side-effect.

        Creates a new result object using the current fitted parameters,
        applied to a completely new dataset that is assumed to be unrelated to the model’s original data.
        The new results can then be used for analysis or forecasting.
        It should be used before forecasting, to wedge the historic data just before the first forecast timestamp,
        with:
            - New observations from the modeled time-series process.
            - New observations of exogenous regressors.

        Parameters
        ----------
        x_past : pd.DataFrame
            The exogenous (features) data.
        y_past : pd.DataFrame
            The endogenous (target) data.

        """
        self.results_ = self.results_.apply(
            endog=y_past,
            exog=x_past,
        )

    def predict_quantile(self, start, end, exog, quantile):
        """Quantile prediction.

        It relies on the parameters' confidence intervals.

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
        predictions = self.results_.predict(start, end, exog=x).to_numpy()
        if quantile != 0.5:
            predictions = self.predict_quantile(start, end, exog=x, quantile=quantile)
        return predictions

    def set_feature_importance(self):
        """Because report needs 'weight' and 'gain' as importance metrics, we set the values to these names.

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

    def score(self, x, y):
        """Compute R2 score with backtesting strategy.

        The backtest  is performed by the Time Series cross-validator of scikit-learn which
        returns first k folds as train set and the (k+1)th fold as test set in the kth split.
        (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

        It needs to update the historic data with (x_past, y_past) for each split.

        """
        ys_true = []
        ys_pred = []

        # Build the cross-validator
        freq = pd.infer_freq(x.index)
        if not (freq[0].isdigit()):
            freq = f"1{freq}"
        max_horizon_delta = pd.Timedelta(self.backtest_max_horizon, "minutes")
        freq_delta = pd.Timedelta(freq)
        test_size = max_horizon_delta // freq_delta
        n_splits = (x.shape[0] // test_size) - 1
        time_series_cross_validator = TimeSeriesSplit(
            n_splits=n_splits, test_size=test_size
        )

        # Backtesting
        for apply_index, test_index in time_series_cross_validator.split(x):
            # Update the historic data to the current split (ie the k first folds)
            updated_results = self.results_.apply(
                y.iloc[apply_index], x.iloc[apply_index]
            )

            # The (k+1)th fold as the test data
            x_test, y_true_test = x.iloc[test_index], y.iloc[test_index]
            start_test = x_test.iloc[0].name
            end_test = x_test.iloc[-1].name

            # Compute and gather the predictions
            y_pred_test = updated_results.predict(
                start=start_test, end=end_test, exog=x_test
            )
            ys_true.append(y_true_test)
            ys_pred.append(y_pred_test)

        ys_true = np.concatenate(ys_true)
        ys_pred = np.concatenate(ys_pred)
        return r2_score(ys_true, ys_pred)
