# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_array


class MissingValuesTransformer:
    in_feature_names: List[str] = None
    _n_in_features: int = None

    non_null_feature_names: List[str] = None

    def __init__(
        self,
        missing_values: Union[int, float, str, None] = np.nan,
        imputation_strategy: str = None,
        fill_value: Union[str, int, float] = None,
    ):
        """Initialize missing values handler."""
        self.missing_values = missing_values
        self.imputation_strategy = imputation_strategy
        self.fill_value = fill_value

    def fit(self, x, y=None):
        _ = check_array(x, force_all_finite="allow-nan")
        if type(x) != pd.DataFrame:
            x = pd.DataFrame(np.asarray(x))

        self.in_feature_names = list(x.columns)
        self._n_in_features = x.shape[1]

        # Remove always null columns
        is_column_null = x.isnull().all(axis="index")
        self.non_null_feature_names = list(x.columns[~is_column_null])

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

        # Imputers do not support labels
        self.imputer_.fit(X=x, y=None)

    def transform(self, x) -> pd.DataFrame:
        _ = check_array(x, force_all_finite="allow-nan")
        if type(x) != pd.DataFrame:
            x = pd.DataFrame(np.asarray(x))

        x = x[self.non_null_feature_names]

        return self.imputer_.transform(x)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    @classmethod
    def _identity(cls, x):
        return x
