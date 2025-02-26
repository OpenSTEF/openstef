# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import math
import re
from typing import Union, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from openstef.feature_engineering.missing_values_transformer import (
    MissingValuesTransformer,
)
from openstef.model.metamodels.feature_clipper import FeatureClipper
from openstef.model.regressors.regressor import OpenstfRegressor

DEFAULT_QUANTILES: tuple[float, ...] = (0.9, 0.5, 0.1)


class GBLinearQuantileOpenstfRegressor(OpenstfRegressor):
    is_fitted_: bool = False

    TO_KEEP_FEATURES: List[str] = [
        "T-7d",
    ]
    TO_IGNORE_FEATURES: List[str] = [
        "Month",
        "Quarter",
    ]

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        missing_values: Union[int, float, str, None] = np.nan,
        imputation_strategy: Optional[str] = "mean",
        fill_value: Union[str, int, float] = None,
        weight_scale_percentile: int = 95,
        weight_exponent: float = 1,
        weight_floor: float = 0.1,
        validation_fraction: float = 0.2,
        no_fill_future_values_features: List[str] = None,
        clipped_features: List[str] = None,
        learning_rate: float = 0.15,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 10,
        reg_alpha: float = 0.0001,
        reg_lambda: float = 0.1,
        updater: str = "shotgun",
        feature_selector: str = "shuffle",
        top_k: int = 0,
    ):
        super().__init__()

        # Check if quantile 0.5 is present. This is required.
        if 0.5 not in quantiles:
            raise ValueError(
                "Cannot train quantile model as 0.5 is not in requested quantiles!"
            )

        if clipped_features is None:
            clipped_features = ["APX"]

        self.quantiles = quantiles
        self.weight_scale_percentile = weight_scale_percentile
        self.weight_exponent = weight_exponent
        self.weight_floor = weight_floor
        self.imputer_ = MissingValuesTransformer(
            missing_values=missing_values,
            imputation_strategy=imputation_strategy,
            fill_value=fill_value,
            no_fill_future_values_features=no_fill_future_values_features,
        )
        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()
        self.validation_fraction = validation_fraction
        self.model_: xgb.Booster = None
        self.feature_clipper_ = FeatureClipper(columns=clipped_features)

        self.learning_rate = learning_rate
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.reg_alpha = reg_alpha
        self.reg_labmda = reg_lambda
        self.updater = updater
        self.feature_selector = feature_selector
        self.top_k = top_k

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

        if feature_name in self.TO_KEEP_FEATURES:
            return False

        return (
            # Ignore named features
            feature_name in self.TO_IGNORE_FEATURES
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

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs) -> OpenstfRegressor:
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y), name="load")

        x = self._remove_ignored_features(x)
        self.feature_clipper_.fit(x)

        # Fix nan columns
        x, y = self.imputer_.fit_transform(x, y)
        if x.isna().any().any():
            raise ValueError(
                "There are nan values in the input data. Set "
                "imputation_strategy to solve them."
            )

        # Apply feature scaling
        x_scaled = self.x_scaler_.fit_transform(x)
        y_scaled = self.y_scaler_.fit_transform(y.to_frame())[:, 0]

        # Add more focus on extreme / peak values
        sample_weight = self._calculate_sample_weights(y.values.squeeze())

        # Split the data into training and validation sets
        x_train, x_val, y_train, y_val, weight_train, weight_val = train_test_split(
            x_scaled,
            y_scaled,
            sample_weight,
            test_size=self.validation_fraction,
            random_state=42,
        )

        # Preserve feature names
        x_train = pd.DataFrame(x_train, columns=x.columns)
        x_val = pd.DataFrame(x_val, columns=x.columns)

        dtrain = xgb.DMatrix(x_train, label=y_train, weight=weight_train)
        dval = xgb.DMatrix(x_val, label=y_val, weight=weight_val)

        xgb_params = {
            # Use the quantile objective function.
            "objective": "reg:quantileerror",  # This is pinball loss
            "booster": "gblinear",
            "updater": self.updater,
            "alpha": self.reg_alpha,
            "lambda": self.reg_labmda,
            "feature_selector": self.feature_selector,
            "quantile_alpha": np.array(self.quantiles),
            "learning_rate": self.learning_rate,
        }

        if self.top_k > 0:
            xgb_params["top_k"] = self.top_k

        self.model_ = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dtrain, "train"), (dval, "val")],
        )

        self._Booster = self.model_

        self.is_fitted_ = True

        self.feature_importances_ = self._get_feature_importances_from_booster(
            self.model_
        )

        return self

    def _calculate_sample_weights(self, y: np.array):
        """Calculate sample weights based on the y values of arbitrary scale.

        The resulting weights are in the range [0,1] and are used to put more emphasis
        on certain samples. The sample weighting function does:

        * Rescale data to a [-1, 1] range using quantile scaling. 90% of the data will
          be within this range. Rest is outside.
        * Calculate the weight by taking the exponent of scaled data.
          * exponent=0: Results in uniform weights for all samples.
          * exponent=1: Results in linearly increasing weights for samples that are
            closer to the extremes.
          * exponent>1: Results in exponentially increasing weights for samples that are
            closer to the extremes.
        * Clip the data to [0, 1] range with weight_floor as the minimum weight.
          * Weight floor is used to make sure that all the samples are considered.

        """
        return np.clip(
            _weight_exp(
                _scale_percentile(y, percentile=self.weight_scale_percentile),
                exponent=self.weight_exponent,
            ),
            a_min=self.weight_floor,
            a_max=1,
        )

    def predict(self, x: pd.DataFrame, quantile: float = 0.5, **kwargs) -> np.array:
        check_is_fitted(self)

        # Preprocess input data
        x = self._remove_ignored_features(x)
        x = self.feature_clipper_.transform(x)
        x = self.imputer_.transform(x)
        x_scaled = self.x_scaler_.transform(x)

        # Preserve feature names
        x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

        d_x_scaled = xgb.DMatrix(x_scaled)

        # Make prediction
        y_pred = self.model_.predict(d_x_scaled)

        # When multiple quantiles are trained,
        # we need to select the requested quantile
        if len(self.quantiles) > 1:
            # Get index of the quantile value in the quantiles list
            quantile_index = self.quantiles.index(quantile)

            # Get the quantile prediction
            y_pred = y_pred[:, quantile_index]

        # Inverse scaling
        y_pred = self.y_scaler_.inverse_transform(y_pred.reshape(-1, 1))[:, 0]

        return y_pred

    @classmethod
    def _get_feature_importances_from_booster(cls, booster: xgb.Booster) -> np.ndarray:
        """Gets feature importances from a XGB booster.

        This is based on the feature_importance_ property defined in:
        https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py.

        Args:
            booster: Booster object,
                most of the times the median model (quantile=0.5) is preferred

        Returns:
            Ndarray with normalized feature importances.

        """
        # Get score
        score = booster.get_score(importance_type="weight")

        if type(next(iter(score.values()))) is list:
            num_quantiles = len(next(iter(score.values())))

            # Select middle quantile, assuming odd number of quantiles
            quantile_index = num_quantiles // 2

            score = {f: score[f][quantile_index] for f in score}

        # Get feature names from booster
        feature_names = booster.feature_names

        # Get importance
        feature_importance = [np.abs(score.get(f, 0.0)) for f in feature_names]
        # Convert to array
        features_importance_array = np.array(feature_importance, dtype=np.float32)

        total = features_importance_array.sum()  # For normalizing
        if total == 0:
            return features_importance_array
        return features_importance_array / total  # Normalize

    @classmethod
    def _get_param_names(cls):
        return [
            "quantiles",
        ]

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted_


def _scale_percentile(x: np.ndarray, percentile: int = 95):
    return np.abs(x / np.percentile(np.abs(x), percentile))


def _weight_exp(x: np.ndarray, exponent: float = 1):
    return np.abs(x) ** exponent
