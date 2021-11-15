# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class MissingValueHandler(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Class for a meta-models that handles missing values and removes columns filled exclusively by NaN.
    Pipeline of imputer for missing value and a regressor.

    Parameters
    ----------

    base_estimator: RegressorMixin
        Regressor used in the pipeline.

    missing_values: default=np.nan
        Type of missing values handled by the imputer.

    imputation_strategy: str, default=None
        Imputing strategy for the imputer. If None, nan values are not allow and the linear regression will raise a ValueError.

    fill_value: optional
        Value used by the imputer for missing values.

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
        tags["multioutput"] = False
        tags["allow_nan"] = self.imputation_strategy is not None
        return tags

    def fit(self, x, y):

        _, y = check_X_y(x, y, force_all_finite="allow-nan", y_numeric=True)
        if type(x) != pd.DataFrame:
            x = pd.DataFrame(np.asarray(x))

        # Remove always null columns
        columns = x.isnull().all(0)
        self.feature_names_ = list(x.columns)
        self.non_null_columns_ = list(columns[~columns].index)
        self.n_features_in_ = x.shape[1]

        self.regressor_ = clone(self.base_estimator)
        if self.imputation_strategy is None:
            self.imputer_ = None
            self.pipeline_ = Pipeline([("regressor", self.regressor_)])
        else:
            self.imputer_ = SimpleImputer(
                missing_values=self.missing_values,
                strategy=self.imputation_strategy,
                fill_value=self.fill_value,
            )
            self.pipeline_ = Pipeline(
                [("imputer", self.imputer_), ("regressor", self.regressor_)]
            )

        self.pipeline_.fit(x[self.non_null_columns_], y)

        if hasattr(self.regressor_, "feature_importances_"):
            reg_feature_importances = self.regressor_.feature_importances_
            self.feature_importances_ = np.zeros(self.n_features_in_)
            j = 0
            for i, c in enumerate(self.feature_names_):
                if c in self.non_null_columns_:
                    self.feature_importances_[i] = reg_feature_importances[j]
                    j += 1

        return self

    def predict(self, x):
        check_is_fitted(self)
        check_array(
            x,
            force_all_finite="allow-nan",
        )
        if type(x) != pd.DataFrame:
            x = pd.DataFrame(np.array(x))
        return self.pipeline_.predict(x[self.non_null_columns_])
