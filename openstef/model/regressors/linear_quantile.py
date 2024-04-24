# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import re
from typing import Dict, Union, Set, Optional

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from openstef.feature_engineering.missing_values_transformer import (
    MissingValuesTransformer,
)
from openstef.model.regressors.regressor import OpenstfRegressor

DEFAULT_QUANTILES: tuple[float, ...] = (0.9, 0.5, 0.1)


class LinearQuantileOpenstfRegressor(OpenstfRegressor, RegressorMixin):
    quantiles: tuple[float, ...]
    alpha: float
    solver: str

    imputer_: MissingValuesTransformer
    x_scaler_: MinMaxScaler
    y_scaler_: MinMaxScaler
    models_: Dict[float, QuantileRegressor]

    is_fitted_: bool = False

    FEATURE_IGNORE_LIST: Set[str] = {
        "IsWeekendDay",
        "IsWeekDay",
        "IsSunday",
        "Month",
        "Quarter",
    }

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        alpha: float = 0.0,
        solver: str = "highs",
        missing_values: Union[int, float, str, None] = np.nan,
        imputation_strategy: Optional[str] = "mean",
        fill_value: Union[str, int, float] = None,
    ):
        """Initialize LinearQuantileOpenstfRegressor.

        Model that provides quantile regression with SKLearn QuantileRegressor.
        For each desired quantile an QuantileRegressor model is trained,
        these can later be used to predict quantiles.

        This model is sensitive to feature quality and therefore has logic to remove
        some custom features produced by OpenSTEF. The features that are removed are:
        - Holiday features (is_christmas, is_*)
        - Lagged features (T-1d, T-*)
        - Point in time features (IsWeekendDay, IsWeekDay, IsSunday, Month, Quarter)
        - Infeed MFFBAS profiles (E*_I)

        Args:
            quantiles: Tuple with desired quantiles, quantile 0.5 is required.
                For example: (0.1, 0.5, 0.9)
            alpha: Regularization constant for L1 regularization
            solver: Solver to use for optimization
            missing_values: Value to be considered as missing value
            imputation_strategy: Imputation strategy
            fill_value: Fill value

        """
        super().__init__()

        # Check if quantile 0.5 is present. This is required.
        if 0.5 not in quantiles:
            raise ValueError(
                "Cannot train quantile model as 0.5 is not in requested quantiles!"
            )

        self.quantiles = quantiles
        self.alpha = alpha
        self.solver = solver
        self.imputer_ = MissingValuesTransformer(
            missing_values=missing_values,
            imputation_strategy=imputation_strategy,
            fill_value=fill_value,
        )
        self.x_scaler_ = MinMaxScaler(feature_range=(-1, 1))
        self.y_scaler_ = MinMaxScaler(feature_range=(-1, 1))
        self.models_ = {
            quantile: QuantileRegressor(alpha=alpha, quantile=quantile, solver=solver)
            for quantile in quantiles
        }

    @property
    def feature_names(self) -> list:
        """The names of the features used to train the model."""
        check_is_fitted(self)
        return self.imputer_.non_null_feature_names

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }

    @property
    def can_predict_quantiles(self) -> bool:
        """Attribute that indicates if the model predict particular quantiles."""
        return True

    def _is_feature_ignored(self, feature_name: str) -> bool:
        """Check if a feature is ignored by the model.

        Args:
            feature_name: Feature name

        Returns:
            True if the feature is ignored, False otherwise

        """
        return (
            # Ignore named features
            feature_name in self.FEATURE_IGNORE_LIST
            or
            # Ignore holiday features
            re.match(r"is_", feature_name) is not None
            or
            # Ignore lag features
            re.match(r"T-", feature_name) is not None
            or
            # Ignore infeed MFFBAS profiles
            re.match(r"E\d.*_I", feature_name) is not None
        )

    def _remove_ignored_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Remove ignored features from the input data.

        Args:
            x: Input data

        Returns:
            Data without ignored features

        """
        return x.drop(columns=[c for c in x.columns if self._is_feature_ignored(c)])

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs) -> RegressorMixin:
        """Fits linear quantile model.

        Args:
            x: Feature matrix
            y: Labels

        Returns:
            Fitted LinearQuantile model

        """
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y), name="load")

        x = self._remove_ignored_features(x)

        # Fix nan columns
        x = self.imputer_.fit_transform(x)
        if x.isna().any().any():
            raise ValueError(
                "There are nan values in the input data. Set "
                "imputation_strategy to solve them."
            )

        # Apply feature scaling
        x_scaled = self.x_scaler_.fit_transform(x)
        y_scaled = self.y_scaler_.fit_transform(y.to_frame())[:, 0]

        # Add more focus on extreme / peak values
        sample_weight = np.abs(y_scaled)

        # Fit quantile regressors
        for quantile in self.quantiles:
            self.models_[quantile].fit(
                X=x_scaled, y=y_scaled, sample_weight=sample_weight
            )

        self.is_fitted_ = True

        self.feature_importances_ = self._get_feature_importance_from_linear()

        return self

    def predict(self, x: pd.DataFrame, quantile: float = 0.5, **kwargs) -> np.array:
        """Makes a prediction for a desired quantile.

        Args:
            x: Feature matrix
            quantile: Quantile for which a prediciton is desired,
                note that only quantile are available for which a model is trained,
                and that this is a quantile-model specific keyword

        Returns:
            Prediction

        Raises:
            ValueError in case no model is trained for the requested quantile

        """
        check_is_fitted(self)

        # Preprocess input data
        x = self._remove_ignored_features(x)
        x = self.imputer_.transform(x)
        x_scaled = self.x_scaler_.transform(x)

        # Make prediction
        y_pred = self.models_[quantile].predict(X=x_scaled)

        # Inverse scaling
        y_pred = self.y_scaler_.inverse_transform(y_pred.reshape(-1, 1))[:, 0]

        return y_pred

    def _get_feature_importance_from_linear(self, quantile: float = 0.5) -> np.array:
        check_is_fitted(self)
        feature_importance_linear = np.abs(self.models_[quantile].coef_)
        reg_feature_importances_dict = dict(
            zip(self.imputer_.non_null_feature_names, feature_importance_linear)
        )
        return np.array(
            [
                reg_feature_importances_dict.get(c, 0)
                for c in self.imputer_.in_feature_names
            ]
        )

    @classmethod
    def _get_param_names(cls):
        return [
            "quantiles",
            "alpha",
            "solver",
        ]

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted_
