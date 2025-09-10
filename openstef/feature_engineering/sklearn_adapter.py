# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501
#
# SPDX-License-Identifier: MPL-2.0

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SklearnTransformerAdapter(BaseEstimator, TransformerMixin):
    """Adapter to use any sklearn transformer in OpenSTEF pipelines.

    This adapter wraps any scikit-learn transformer and makes it compatible
    with OpenSTEF's pipeline system while maintaining the sklearn interface.

    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> adapter = SklearnTransformerAdapter(StandardScaler)
        >>> X_transformed = adapter.fit_transform(X_train)

    Example with custom parameters:
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> adapter = SklearnTransformerAdapter(MinMaxScaler, feature_range=(0, 1))
        >>> X_scaled = adapter.fit_transform(X_train)
    """

    def __init__(self, sklearn_transformer: type, **kwargs):
        """Initialize the sklearn transformer adapter.

        Args:
            sklearn_transformer: The sklearn transformer class to adapt
            **kwargs: Additional arguments to pass to the sklearn transformer
        """
        self.sklearn_transformer = sklearn_transformer
        self.transformer_kwargs = kwargs
        self.transformer_ = None
        self.is_fitted_ = False  # Follow OpenSTEF pattern

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the underlying sklearn transformer.

        Args:
            X: Input features as pandas DataFrame
            y: Target values (optional)

        Returns:
            self: Fitted transformer adapter
        """
        # Initialize the sklearn transformer with provided kwargs
        self.transformer_ = self.sklearn_transformer(**self.transformer_kwargs)

        # Fit the transformer
        self.transformer_.fit(X, y)

        self.is_fitted_ = True  # Follow OpenSTEF pattern
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted sklearn transformer.

        Args:
            X: Input features to transform

        Returns:
            Transformed features as pandas DataFrame

        Raises:
            ValueError: If transformer is not fitted yet
        """
        # Check if transformer is fitted
        if self.transformer_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        # Use sklearn's validation (like other OpenSTEF components)
        check_is_fitted(self.transformer_)

        # Transform the data
        transformed = self.transformer_.transform(X)

        # Ensure output is pandas DataFrame (maintain OpenSTEF interface)
        if not isinstance(transformed, pd.DataFrame):
            # Try to preserve column names and index
            if hasattr(X, 'columns') and transformed.shape[1] == len(X.columns):
                transformed = pd.DataFrame(
                    transformed, columns=X.columns, index=X.index
                )
            else:
                transformed = pd.DataFrame(transformed, index=X.index)

        return transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit and transform in one step.

        Args:
            X: Input features
            y: Target values (optional)

        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        """Check if the adapter is fitted.

        Follows OpenSTEF pattern seen in MissingValuesTransformer.
        """
        return self.is_fitted_