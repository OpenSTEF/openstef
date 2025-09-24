# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

from openstef.feature_engineering.sklearn_adapter import SklearnTransformerAdapter


class TestSklearnTransformerAdapter:
    """Test suite for SklearnTransformerAdapter."""

    def setup_method(self):
        """Set up test data."""
        self.X = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5], 
            "feature2": [10, 20, 30, 40, 50]
        }, index=pd.date_range('2023-01-01', periods=5, freq='h'))

    def test_fit_and_transform_preserves_index_and_columns(self):
        """Test that fit/transform preserves DataFrame structure."""
        adapter = SklearnTransformerAdapter(StandardScaler)
        adapter.fit(self.X)
        result = adapter.transform(self.X)
        
        # Check structure preservation
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.X.shape
        assert list(result.columns) == list(self.X.columns)
        assert result.index.equals(self.X.index)

    def test_fit_transform_with_parameters(self):
        """Test adapter works with transformer parameters."""
        adapter = SklearnTransformerAdapter(MinMaxScaler, feature_range=(0, 1))
        result = adapter.fit_transform(self.X)
        
        assert isinstance(result, pd.DataFrame)
        # Check that scaling worked (values should be between 0 and 1)
        assert result.min().min() >= 0
        assert result.max().max() <= 1

    def test_different_sklearn_transformers(self):
        """Test adapter works with various sklearn transformers."""
        transformers_to_test = [
            StandardScaler,
            MinMaxScaler,
            RobustScaler,
        ]
        
        for transformer_class in transformers_to_test:
            adapter = SklearnTransformerAdapter(transformer_class)
            result = adapter.fit_transform(self.X)
            assert isinstance(result, pd.DataFrame)
            assert not result.isnull().any().any()  # No NaN values

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        adapter = SklearnTransformerAdapter(StandardScaler)
        
        with pytest.raises(ValueError):
            adapter.transform(self.X)

    def test_is_fitted_method(self):
        """Test the __sklearn_is_fitted__ method."""
        adapter = SklearnTransformerAdapter(StandardScaler)
        assert not adapter.__sklearn_is_fitted__()
        
        adapter.fit(self.X)
        assert adapter.__sklearn_is_fitted__()

    def test_with_missing_values(self):
        """Test adapter works with missing values."""
        X_with_nan = self.X.copy()
        X_with_nan.loc['2023-01-01 02:00:00', 'feature1'] = np.nan
        
        adapter = SklearnTransformerAdapter(SimpleImputer, strategy='mean')
        result = adapter.fit_transform(X_with_nan)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.isnull().any().any()  # No NaN values after imputation