# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureStats(BaseModel):
    mean: float
    std: float


class FeatureClipper(BaseEstimator, TransformerMixin):
    """
    A transformer that clips the values of specified columns to the minimum and
    maximum values observed during training. This prevents the model from
    extrapolating beyond these values during prediction.
    """

    def __init__(self, columns: List[str], clip_number_of_std: float = 2.0):
        """
        Initialize the FeatureClipper.

        Parameters:
        ----------
        columns : List[str]
            List of column names to be clipped.
        """
        self.columns: List[str] = columns
        self.feature_ranges: Dict[str, Tuple[float, float] | FeatureStats] = {}
        self.clip_number_of_std = clip_number_of_std

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureClipper":
        """
        Fits the transformer on the training data by calculating the min and max
        values for the specified columns.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame containing training data.

        y : Optional[pd.Series]
            Ignored. This parameter exists for compatibility with scikit-learn's pipeline.

        Returns:
        -------
        self : FeatureClipper
            Fitted transformer.

        Raises:
        ------
        ValueError:
            If the input is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        for col in self.columns:
            if col in X.columns:
                self.feature_ranges[col] = FeatureStats(
                    mean=X[col].mean(), std=X[col].std()
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data by clipping the specified columns' values to be within
        the min and max range observed during fitting.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame containing new data to be transformed.

        Returns:
        -------
        X_ : pd.DataFrame
            A copy of the input DataFrame with clipped values in the specified columns.

        Raises:
        ------
        ValueError:
            If the input is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        X_copy = X.copy()

        for col in self.columns:
            if col in X_copy.columns and col in self.feature_ranges:
                if isinstance(self.feature_ranges[col], FeatureStats):
                    stats = self.feature_ranges[col]
                    X_copy[col] = X_copy[col].clip(
                        lower=stats.mean - self.clip_number_of_std * stats.std,
                        upper=stats.mean + self.clip_number_of_std * stats.std,
                    )
                else:
                    # Backward compatibility with previous minmax-based implementation
                    min_val, max_val = self.feature_ranges[col]
                    X_copy[col] = X_copy[col].clip(lower=min_val, upper=max_val)

        return X_copy
