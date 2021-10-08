import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from openstf.model.regressors.regressor import OpenstfRegressor


class LinearRTEOpenstfRegressor(OpenstfRegressor):
    """Class for Linear Models that handles missing values,

    Pipeline of imputer for missing value and linear regressor.

    Parameters
    ----------
    missing_values: default=np.nan
        Type of missing values handled by the imputer.

    imputation_strategy: str, default=None
        Imputing strategy for the imputer. If None, nan values are not allow and the linear regression will raise a ValueError.

    fill_value: optional
        Value used by the imputer for missing values.

     Attributes
    ----------
        feature_names_: list(str)
            All input feature.

        non_null_columns_: list(str)
            Valid features used by the regressor.

        n_features_in_: int
            Number of input features.

        regressor_: LinearRegression
            Linear regressor on valid columns.

        imputer_: SimpleImputer
            Imputer for missig value.

        pipeline_: Pipeline
            Pipeline that chains imputer and linear regressor.

        feature_importances_: ndarray (n_features_in_, )
            The absolute values of the regressions' coefficients for valid feratures and zero otherwise.
    """

    gain_importance_name = "total_gain"
    weight_importance_name = "weight"

    def __init__(
        self, missing_values=np.nan, imputation_strategy=None, fill_value=None
    ):
        self.missing_values = missing_values
        self.imputation_strategy = imputation_strategy
        self.fill_value = fill_value

    def fit(self, X, y, **kwargs):
        _, y = check_X_y(X, y, force_all_finite="allow-nan", y_numeric=True)
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(np.asarray(X))

        # Remove always null columns
        columns = X.isnull().all(0)
        self.feature_names_ = list(X.columns)
        self.non_null_columns_ = list(columns[~columns].index)
        self.n_features_in_ = X.shape[1]

        self.regressor_ = LinearRegression()
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

        self.pipeline_.fit(X[self.non_null_columns_], y)

        reg_feature_importances = np.abs(self.regressor_.coef_)
        self.feature_importances_ = np.zeros(len(self.feature_names_))
        j = 0
        for i, c in enumerate(self.feature_names_):
            if c in self.non_null_columns_:
                self.feature_importances_[i] = reg_feature_importances[j]
                j += 1

        return self

    def predict(self, X):
        check_is_fitted(self)
        check_array(
            X,
            force_all_finite="allow-nan",
        )
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(np.array(X))
        return self.pipeline_.predict(X[self.non_null_columns_])
