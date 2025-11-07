"""Hybrid quantile regression estimators for multi-quantile forecasting.

This module provides the HybridQuantileRegressor class, which combines LightGBM and linear models
using stacking for robust multi-quantile regression, including serialization utilities.
"""

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import QuantileRegressor
from skops.io import dumps, loads
from xgboost import XGBRegressor

from openstef_core.exceptions import ModelLoadingError


class HybridQuantileRegressor:
    """Custom Hybrid regressor for multi-quantile estimation using sample weights."""

    def __init__(  # noqa: D107, PLR0913, PLR0917
        self,
        quantiles: list[float],
        lightgbm_n_estimators: int = 100,
        lightgbm_learning_rate: float = 0.1,
        lightgbm_max_depth: int = -1,
        lightgbm_min_child_weight: float = 1.0,
        ligntgbm_min_child_samples: int = 1,
        lightgbm_min_data_in_leaf: int = 20,
        lightgbm_min_data_in_bin: int = 10,
        lightgbm_reg_alpha: float = 0.0,
        lightgbm_reg_lambda: float = 0.0,
        lightgbm_num_leaves: int = 31,
        lightgbm_max_bin: int = 255,
        lightgbm_subsample: float = 1.0,
        lightgbm_colsample_by_tree: float = 1.0,
        lightgbm_colsample_by_node: float = 1.0,
        gblinear_n_steps: int = 100,
        gblinear_learning_rate: float = 0.15,
        gblinear_reg_alpha: float = 0.0001,
        gblinear_reg_lambda: float = 0,
        gblinear_feature_selector: str = "shuffle",
        gblinear_updater: str = "shotgun",
    ):
        self.quantiles = quantiles

        self._models: list[StackingRegressor] = []

        for q in quantiles:
            lightgbm_model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                min_child_samples=ligntgbm_min_child_samples,
                n_estimators=lightgbm_n_estimators,
                learning_rate=lightgbm_learning_rate,
                max_depth=lightgbm_max_depth,
                min_child_weight=lightgbm_min_child_weight,
                min_data_in_leaf=lightgbm_min_data_in_leaf,
                min_data_in_bin=lightgbm_min_data_in_bin,
                reg_alpha=lightgbm_reg_alpha,
                reg_lambda=lightgbm_reg_lambda,
                num_leaves=lightgbm_num_leaves,
                max_bin=lightgbm_max_bin,
                subsample=lightgbm_subsample,
                colsample_bytree=lightgbm_colsample_by_tree,
                colsample_bynode=lightgbm_colsample_by_node,
                verbosity=-1,
                linear_tree=False,
            )

            linear = XGBRegressor(
                booster="gblinear",
                # Core parameters for forecasting
                objective="reg:quantileerror",
                n_estimators=gblinear_n_steps,
                learning_rate=gblinear_learning_rate,
                # Regularization parameters
                reg_alpha=gblinear_reg_alpha,
                reg_lambda=gblinear_reg_lambda,
                # Boosting structure control
                feature_selector=gblinear_feature_selector,
                updater=gblinear_updater,
                quantile_alpha=q,
            )

            final_estimator = QuantileRegressor(quantile=q)

            self._models.append(
                StackingRegressor(
                    estimators=[("lightgbm", lightgbm_model), ("gblinear", linear)],  # type: ignore
                    final_estimator=final_estimator,
                    verbose=3,
                    passthrough=False,
                    n_jobs=None,
                    cv=2,
                )
            )
        self.is_fitted: bool = False
        self.feature_names: list[str] = []

    def fit(
        self,
        X: npt.NDArray[np.floating] | pd.DataFrame,  # noqa: N803
        y: npt.NDArray[np.floating] | pd.Series,
        sample_weight: npt.NDArray[np.floating] | pd.Series | None = None,
        feature_name: list[str] | None = None,
    ) -> None:
        """Fit the multi-quantile regressor.

        Args:
            X: Input features as a DataFrame.
            y: Target values as a 2D array where each column corresponds to a quantile.
            sample_weight: Sample weights for training data.
            feature_name: List of feature names.
        """
        self.feature_names = feature_name if feature_name is not None else []

        X = X.ffill().fillna(0)  # type: ignore
        for model in self._models:
            model.fit(
                X=X,  # type: ignore
                y=y,
                sample_weight=sample_weight,
            )
        self.is_fitted = True

    def predict(self, X: npt.NDArray[np.floating] | pd.DataFrame) -> npt.NDArray[np.floating]:  # noqa: N803
        """Predict quantiles for the input features.

        Args:
            X: Input features as a DataFrame.

        Returns:

            A 2D array where each column corresponds to predicted quantiles.
        """  # noqa: D412
        X = X.ffill().fillna(0)  # type: ignore
        return np.column_stack([model.predict(X=X) for model in self._models])  # type: ignore

    def save_bytes(self) -> bytes:
        """Serialize the model.

        Returns:
            A string representation of the model.
        """
        return dumps(self)

    @classmethod
    def load_bytes(cls, model_bytes: bytes) -> Self:
        """Deserialize the model from bytes using joblib.

        Args:
            model_bytes : Bytes representing the serialized model.

        Returns:
            An instance of LightGBMQuantileRegressor.

        Raises:
            ModelLoadingError: If the deserialized object is not a HybridQuantileRegressor.
        """
        trusted_types = [
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMRegressor",
            "sklearn.utils._bunch.Bunch",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor",
            "openstef_models.estimators.hybrid.HybridQuantileRegressor",
        ]
        instance = loads(model_bytes, trusted=trusted_types)

        if not isinstance(instance, cls):
            raise ModelLoadingError("Deserialized object is not a HybridQuantileRegressor")

        return instance
