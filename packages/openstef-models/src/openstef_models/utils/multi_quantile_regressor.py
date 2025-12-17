# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Adaptor for multi-quantile regression using a base quantile regressor.

Designed to work with scikit-learn compatible regressors that support quantile regression.
"""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

ParamType = float | int | str | bool | None


class MultiQuantileRegressor(BaseEstimator, RegressorMixin):
    """Adaptor for multi-quantile regression using a base quantile regressor.

    This class creates separate instances of a given quantile regressor for each quantile
    and manages their training and prediction.
    """

    def __init__(
        self,
        base_learner: type[BaseEstimator],
        quantile_param: str,
        quantiles: list[float],
        hyperparams: dict[str, ParamType],
    ):
        """Initialize MultiQuantileRegressor.

        This is an adaptor that allows any quantile-capable regressor to predict multiple quantiles
        by instantiating separate models for each quantile.

        Args:
            base_learner: A scikit-learn compatible regressor class that supports quantile regression.
            quantile_param: The name of the parameter in base_learner that sets the quantile level.
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
            hyperparams: Dictionary of hyperparameters to pass to each base learner instance.
        """
        self.quantiles = quantiles
        self.hyperparams = hyperparams
        self.quantile_param = quantile_param
        self.base_learner = base_learner
        self.is_fitted = False
        self._models = [self._init_model(q) for q in quantiles]

    def _init_model(self, q: float) -> BaseEstimator:
        params = self.hyperparams.copy()
        params[self.quantile_param] = q
        base_learner = self.base_learner(**params)

        if self.quantile_param not in base_learner.get_params():  # type: ignore
            msg = f"The base learner does not support the quantile parameter '{self.quantile_param}'."
            raise ValueError(msg)

        return base_learner

    def fit(
        self,
        X: npt.NDArray[np.floating] | pd.DataFrame,
        y: npt.NDArray[np.floating] | pd.Series,
        sample_weight: npt.NDArray[np.floating] | pd.Series | None = None,
        feature_name: list[str] | None = None,
        eval_set: list[tuple[pd.DataFrame, npt.NDArray[np.floating]]] | None = None,
        eval_sample_weight: list[npt.NDArray[np.floating]]
        | list[pd.Series]
        | None = None,
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
        # Pass model-specific eval arguments
        kwargs = {}
        for model in self._models:
            # Check if early stopping is supported
            # Check that eval_set is supported
            if eval_set is None and "early_stopping_rounds" in self.hyperparams:
                model.set_params(early_stopping_rounds=None)  # type: ignore

            if (
                eval_set is not None
                and self.learner_eval_sample_weight_param is not None
            ):  # type: ignore
                kwargs[self.learner_eval_sample_weight_param] = eval_sample_weight

            if (
                "early_stopping_rounds" in self.hyperparams
                and self.learner_eval_sample_weight_param is not None
            ):
                model.set_params(
                    early_stopping_rounds=self.hyperparams["early_stopping_rounds"]
                )  # type: ignore

            if feature_name:
                self.model_feature_names = feature_name
            else:
                self.model_feature_names = []

            if eval_sample_weight is not None and self.learner_eval_sample_weight_param:
                kwargs[self.learner_eval_sample_weight_param] = eval_sample_weight

            model.fit(  # type: ignore
                X=np.asarray(X),
                y=y,
                sample_weight=sample_weight,
                **kwargs,
            )

        self.is_fitted = True

    @property
    def learner_eval_sample_weight_param(self) -> str | None:
        """Get the name of the sample weight parameter for evaluation sets.

        Returns:
            The name of the sample weight parameter if supported, else None.
        """
        learner_name: str = self.base_learner.__name__
        params: dict[str, str | None] = {
            "QuantileRegressor": None,
            "LGBMRegressor": "eval_sample_weight",
            "XGBRegressor": "sample_weight_eval_set",
        }
        return params.get(learner_name)

    def predict(
        self, X: npt.NDArray[np.floating] | pd.DataFrame
    ) -> npt.NDArray[np.floating]:
        """Predict quantiles for the input features.

        Args:
            X: Input features as a DataFrame.

        Returns:

            A 2D array where each column corresponds to predicted quantiles.
        """  # noqa: D412
        return np.column_stack([model.predict(X=X) for model in self._models])  # type: ignore

    @property
    def models(self) -> list[BaseEstimator]:
        """Get the list of underlying quantile models.

        Returns:
            List of BaseEstimator instances for each quantile.
        """
        return self._models

    @property
    def has_feature_names(self) -> bool:
        """Check if the base learners have feature names.

        Returns:
            True if the base learners have feature names, False otherwise.
        """
        return len(self.model_feature_names) > 0
