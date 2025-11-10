# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Custom LightGBM regressor for multi-quantile regression.

This module provides the LGBMQuantileRegressor class, which extends LightGBM's LGBMRegressor
to support multi-quantile output by configuring the objective function accordingly. Each quantile is predicted
by a separate tree within the same boosting ensemble. The module also includes serialization utilities.
"""

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from skops.io import dumps, loads

from openstef_core.exceptions import ModelLoadingError


class LGBMQuantileRegressor(BaseEstimator, RegressorMixin):
    """Custom LightGBM regressor for multi-quantile regression.

    Extends LGBMRegressor to support multi-quantile output by configuring
    the objective function accordingly. Each quantile is predicted by a
    separate tree within the same boosting ensemble.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        quantiles: list[float],
        linear_tree: bool,  # noqa: FBT001
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        min_child_weight: float = 1.0,
        min_data_in_leaf: int = 20,
        min_data_in_bin: int = 10,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        num_leaves: int = 31,
        max_bin: int = 255,
        colsample_bytree: float = 1.0,
        random_state: int | None = None,
        early_stopping_rounds: int | None = None,
        verbosity: int = -1,
    ) -> None:
        """Initialize LgbLinearQuantileRegressor with quantiles.

        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
            linear_tree: Whether to use linear trees.
            n_estimators: Number of boosting rounds/trees to fit.
            learning_rate: Step size shrinkage used to prevent overfitting.
            max_depth: Maximum depth of trees.
            min_child_weight: Minimum sum of instance weight (hessian) needed in a child.
            min_data_in_leaf: Minimum number of data points in a leaf.
            min_data_in_bin: Minimum number of data points in a bin.
            reg_alpha: L1 regularization on leaf weights.
            reg_lambda: L2 regularization on leaf weights.
            num_leaves: Maximum number of leaves.
            max_bin: Maximum number of discrete bins for continuous features.
            colsample_bytree: Fraction of features used when constructing each tree.
            random_state: Random seed for reproducibility.
            early_stopping_rounds: Training will stop if performance doesn't improve for this many rounds.
            verbosity: Verbosity level for LgbLinear training.

        """
        self.quantiles = quantiles
        self.linear_tree = linear_tree
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_data_in_leaf = min_data_in_leaf
        self.min_data_in_bin = min_data_in_bin
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves
        self.max_bin = max_bin
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbosity = verbosity

        self._models: list[LGBMRegressor] = [
            LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                min_data_in_leaf=min_data_in_leaf,
                min_data_in_bin=min_data_in_bin,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                num_leaves=num_leaves,
                max_bin=max_bin,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                early_stopping_rounds=early_stopping_rounds,
                verbosity=verbosity,
                linear_tree=linear_tree,
            )
            for q in quantiles  # type: ignore
        ]

    def fit(
        self,
        X: npt.NDArray[np.floating] | pd.DataFrame,
        y: npt.NDArray[np.floating] | pd.Series,
        sample_weight: npt.NDArray[np.floating] | pd.Series | None = None,
        feature_name: list[str] | None = None,
        eval_set: list[tuple[pd.DataFrame, npt.NDArray[np.floating]]] | None = None,
        eval_sample_weight: list[npt.NDArray[np.floating]] | None = None,
    ) -> None:
        """Fit the multi-quantile regressor.

        Args:
            X: Input features as a DataFrame.
            y: Target values as a 2D array where each column corresponds to a quantile.
            sample_weight: Sample weights for training data.
            feature_name: List of feature names.
            eval_set: Evaluation set for early stopping.
            eval_sample_weight: Sample weights for evaluation data.
        """
        for model in self._models:
            if eval_set is None:
                model.set_params(early_stopping_rounds=None)
            else:
                model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            model.fit(  # type: ignore
                X=np.asarray(X),
                y=y,
                eval_metric="quantile",
                sample_weight=sample_weight,
                eval_set=eval_set,  # type: ignore
                eval_sample_weight=eval_sample_weight,  # type: ignore
                feature_name=feature_name,  # type: ignore
            )

    def predict(self, X: npt.NDArray[np.floating] | pd.DataFrame) -> npt.NDArray[np.floating]:
        """Predict quantiles for the input features.

        Args:
            X: Input features as a DataFrame.

        Returns:

            A 2D array where each column corresponds to predicted quantiles.
        """  # noqa: D412
        return np.column_stack([model.predict(X=np.asarray(X)) for model in self._models])  # type: ignore

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201
        """Check if all models are fitted.

        Returns:
            True if all quantile models are fitted, False otherwise.
        """
        return all(model.__sklearn_is_fitted__() for model in self._models)

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
            An instance of LgbLinearQuantileRegressor.

        Raises:
            ModelLoadingError: If the deserialized object is not a LgbLinearQuantileRegressor.
        """
        trusted_types = [
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMRegressor",
            "openstef_models.estimators.lightgbm.LGBMQuantileRegressor",
        ]
        instance = loads(model_bytes, trusted=trusted_types)

        if not isinstance(instance, cls):
            raise ModelLoadingError("Deserialized object is not a LgbLinearQuantileRegressor")

        return instance

    @property
    def models(self) -> list[LGBMRegressor]:
        """Get the list of underlying quantile models.

        Returns:
            List of LGBMRegressor instances for each quantile.
        """
        return self._models
