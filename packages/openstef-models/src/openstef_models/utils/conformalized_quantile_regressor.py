# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Adaptor for multi-quantile regression using a base quantile regressor.

Designed to work with scikit-learn compatible regressors that support quantile regression.
"""

import logging
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from sklearn.base import BaseEstimator, RegressorMixin

from openstef_core.types import Quantile

logger = logging.getLogger(__name__)

ParamType = float | int | str | bool | None


class ConformalizedQuantileRegressor(BaseEstimator, RegressorMixin):
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
        # Set the quantile parameter to 0.5 as default for the base model
        model_params = {**self.hyperparams, self.quantile_param: 0.5}
        model = self.base_learner(**model_params)
        cv = BlockBootstrap(n_resamplings=10, n_blocks=10, overlapping=False, random_state=59)
        self.model = TimeSeriesRegressor(estimator=cast(RegressorMixin, model), cv=cv, n_jobs=1, method="aci")

    def fit(
        self,
        X: npt.NDArray[np.floating] | pd.DataFrame,
        y: npt.NDArray[np.floating] | pd.Series,
        sample_weight: npt.NDArray[np.floating] | pd.Series | None = None,
    ) -> None:
        """Fit the multi-quantile regressor.

        Args:
            X: Input features as a DataFrame.
            y: Target values as a 2D array where each column corresponds to a quantile.
            sample_weight: Sample weights for training data.
        """
        # Pass model-specific eval arguments
        self.model.fit(X=X, y=y, sample_weight=sample_weight)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict quantiles for the input features.

        Args:
            X: Input features as a DataFrame.

        Returns:
            A DataFrame with predicted quantiles as columns.

        Raises:
            RuntimeError: If the model is not fitted before prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")
        confidences = self.quantiles[: len(self.quantiles) // 2]
        predictions = {c: self._predict_quantile(X, c) for c in confidences}

        df = pd.concat(
            [
                predictions[c].loc[:, [x for x in predictions[c].columns if x != Quantile(0.5).format()]]
                for c in confidences
            ],
            axis=1,
        )

        df[Quantile(0.5).format()] = predictions[confidences[0]][Quantile(0.5).format()]

        return df.sort_index(axis=1)

    def _predict_quantile(
        self,
        X: pd.DataFrame,
        quantile: float,
    ) -> pd.DataFrame:
        """Predict a specific quantile for the input features.

        Args:
            X: Input features as a DataFrame.
            quantile: The quantile level to predict (between 0 and 1).

        Returns:
            A DataFrame with the predicted quantile values.
        """
        q_50, q_quantile = self.model.predict(X=X, confidence_level=quantile, ensemble=True)  # type: ignore
        q_quantile: np.ndarray = q_quantile.reshape(-1, 2)
        df = pd.DataFrame(
            data=q_quantile,
            index=X.index,
            columns=[Quantile(quantile).format(), Quantile(1 - quantile).format()],
        )
        df[Quantile(0.5).format()] = q_50
        return df.sort_index(axis=1)
