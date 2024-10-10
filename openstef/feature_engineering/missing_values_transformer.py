# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_array, check_is_fitted


class MissingValuesTransformer:
    """MissingValuesTransformer handles missing values in data by imputing them with a given strategy.

    It also removes columns that are always null from the data.

    """

    in_feature_names: Optional[List[str]] = None
    _n_in_features: Optional[int] = None

    non_null_feature_names: List[str] = None

    def __init__(
        self,
        missing_values: Union[int, float, str, None] = np.nan,
        imputation_strategy: str = None,
        fill_value: Union[str, int, float] = None,
        no_fill_future_values_features: List[str] = None,
    ):
        """Initialize missing values handler.

        Args:
            missing_values: The placeholder for the missing values. All occurrences of
                `missing_values` will be imputed.
            imputation_strategy: The imputation strategy to use
                Can be one of "mean", "median", "most_frequent", "constant" or None.
            fill_value: When strategy == "constant", fill_value is used to replace all
                occurrences of missing_values.
            no_fill_future_values_features: The features for which it does not make sense
                to fill future values. Rows that contain trailing null values for these
                features will be removed from the data.

        """
        self.missing_values = missing_values
        self.imputation_strategy = imputation_strategy
        self.fill_value = fill_value
        self.no_fill_future_values_features = no_fill_future_values_features or []
        self.is_fitted_ = False

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
            ).set_output(transform="pandas")
            self.imputer_._validate_params()

    @staticmethod
    def _determine_trailing_null_rows(x: pd.DataFrame) -> pd.Series:
        """Determine rows with trailing null values in a DataFrame."""
        return ~x.bfill().isnull().any(axis="columns")

    def fit(self, x, y=None):
        """Fit the imputer on the input data."""
        _ = check_array(x, force_all_finite="allow-nan")
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.asarray(x))

        self.in_feature_names = list(x.columns)
        self._n_in_features = x.shape[1]

        # Remove always null columns
        is_column_null = x.isnull().all(axis="index")
        self.non_null_feature_names = list(x.columns[~is_column_null])
        x = x[self.non_null_feature_names]

        # Remove trailing null rows for features that should
        # not be imputed in the future
        trailing_null_rows = self._determine_trailing_null_rows(
            x[self.no_fill_future_values_features]
        )
        x = x.loc[trailing_null_rows]

        # Imputers do not support labels
        self.imputer_.fit(X=x, y=None)
        self.is_fitted_ = True

    def transform(self, x) -> pd.DataFrame:
        """Transform the input data by imputing missing values."""
        check_is_fitted(self)
        _ = check_array(x, force_all_finite="allow-nan")
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.asarray(x))

        x = x[self.non_null_feature_names]

        transformed = self.imputer_.transform(x)

        return transformed

    def fit_transform(self, x, y=None) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit the imputer on the input data and transform it.

        Returns:
            The data with missing values imputed.

        """
        self.fit(x, y)

        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.asarray(x))

        x = x[self.non_null_feature_names]

        # Remove trailing null rows for features that should
        # not be imputed in the future
        non_trailing_null_rows = self._determine_trailing_null_rows(
            x[self.no_fill_future_values_features]
        )
        x = x.loc[non_trailing_null_rows]

        x = self.transform(x)

        if y is not None:
            y = y.loc[non_trailing_null_rows]

        return x, y

    @classmethod
    def _identity(cls, x):
        return x

    def __sklearn_is_fitted__(self) -> bool:
        return self.in_feature_names is not None
