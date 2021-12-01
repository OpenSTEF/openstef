# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class MissingValuesHandler(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Class for a meta-models that handles missing values and removes columns filled exclusively by NaN.
    It's a pipeline of:
        - An Imputation transformer for completing missing values.
        - A Regressor fitted on the filled data.

    Parameters
    ----------

    base_estimator: RegressorMixin
        Regressor used in the pipeline.

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

     Attributes
    ----------
        feature_names: list(str)
            All input feature.

        non_null_columns_: list(str)
            Valid features used by the regressor.

        n_features_in_: int
            Number of input features.

        regressor_: RegressorMixin
            Regressor fitted on valid columns.

        imputer_: SimpleImputer
            Imputer for missig value fitted on valid columns.

        pipeline_: Pipeline
            Pipeline that chains the imputer and the regressor.

        feature_importances_: ndarray (n_features_in_, )
            The feature importances from the regressor for valid features and zero otherwise.
    """

    def __init__(
        self,
        base_estimator,
        missing_values=np.nan,
        imputation_strategy=None,
        fill_value=None,
    ):
        self.base_estimator = base_estimator
        self.missing_values = missing_values
        self.imputation_strategy = imputation_strategy
        self.fill_value = fill_value

    def _get_tags(self):
        tags = self.base_estimator._get_tags()
        tags["requires_y"] = True
        tags["multioutput"] = False
        tags["allow_nan"] = self.imputation_strategy is not None
        return tags

    def fit(self, x, y):

        _, y = check_X_y(x, y, force_all_finite="allow-nan", y_numeric=True)
        if type(x) != pd.DataFrame:
            x = pd.DataFrame(np.asarray(x))
        self.feature_in_names_ = list(x.columns)
        self.n_features_in_ = x.shape[1]

        # Remove always null columns
        is_column_null = x.isnull().all(axis="index")
        self.non_null_columns_ = list(x.columns[~is_column_null])

        self.regressor_ = clone(self.base_estimator)

        # Build the proper imputation transformer
        # - Identity function if strategy is None
        # - SimpleImputer with the dedicated strategy
        if self.imputation_strategy is None:
            self.imputer_ = FunctionTransformer(func=self._identity)
        else:
            self.imputer_ = SimpleImputer(
                missing_values=self.missing_values,
                strategy=self.imputation_strategy,
                fill_value=self.fill_value,
            )

        self.pipeline_ = Pipeline(
            [("imputer", self.imputer_), ("regressor", self.regressor_)]
        )

        # Fit only on non_null_columns
        self.pipeline_.fit(x[self.non_null_columns_], y)

        return self

    @classmethod
    def _identity(cls, x):
        return x

    def predict(self, x):
        check_is_fitted(self)
        check_array(
            x,
            force_all_finite="allow-nan",
        )
        if type(x) != pd.DataFrame:
            x = pd.DataFrame(np.array(x))
        return self.pipeline_.predict(x[self.non_null_columns_])
