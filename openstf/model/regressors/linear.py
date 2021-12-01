# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from openstef.model.metamodels.missing_values_handler import MissingValuesHandler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from openstef.model.regressors.regressor import OpenstfRegressor


class LinearRegressor(MissingValuesHandler):
    """Linear Regressor wrapped in the metamodel `MissingValuesHandler` that can handle missing values by imputation strategy.

     Parameters
    ----------

    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    imputation_strategy : str, default=None
        The imputation strategy.
        - If None no imputation is performed.
        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    """

    def __init__(self, missing_values=np.nan, imputation_strategy=None, fill_value=0):
        super().__init__(
            LinearRegression(),
            missing_values=missing_values,
            imputation_strategy=imputation_strategy,
            fill_value=fill_value,
        )


class LinearOpenstfRegressor(LinearRegressor, OpenstfRegressor):
    """Linear Regressor which implements the Openstf regressor API."""

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }

    def fit(self, x, y, **kwargs):
        super().fit(x, y)
        self.feature_importances_ = self._get_feature_importance_from_linear()
        return self

    def _get_feature_importance_from_linear(self):
        check_is_fitted(self)
        feature_importance_linear = np.abs(self.regressor_.coef_)
        reg_feature_importances_dict = dict(
            zip(self.non_null_columns_, feature_importance_linear)
        )
        return np.array(
            [reg_feature_importances_dict.get(c, 0) for c in self.feature_in_names_]
        )

    @property
    def feature_names(self):
        return self.feature_in_names_
