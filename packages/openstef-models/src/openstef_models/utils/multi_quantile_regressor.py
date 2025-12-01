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
from mapie.regression import ConformalizedQuantileRegressor, TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from typing import cast
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

        if self.quantiles == [1-x for x in self.quantiles] and len(self.quantiles)%2 ==1 and self.quantiles[int(len(self.quantiles)/2)]==0.5:
            raise ValueError(
                "Conformalized Quantile Regression requires at least three quantiles and uneven lenght and be symmetrical around 0.5"
            )
        
        # self._models = [self._init_mapie_model(q) for q in alpha]
        self._model = self._init_mapie_model()
        
    def _init_mapie_model(self) -> TimeSeriesRegressor:
        """
        Produces a MAPIE wrapper around the base learner with TimeSeriesRegressor
        """
        base_learner = self._init_base_model(q=0.5)
        cv_mapiets = BlockBootstrap(n_resamplings=3, n_blocks=10)
        mapie_enbpi = TimeSeriesRegressor(cast(RegressorMixin, base_learner), method="enbpi", cv=cv_mapiets, agg_function="mean", n_jobs=-1)
        return mapie_enbpi
        
    def _init_base_model(self, q: float) -> BaseEstimator:
        """
        initialize the base learner with quantile param set to q
        """
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
        eval_sample_weight: list[npt.NDArray[np.floating]] | list[pd.Series] | None = None,
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
        
            # Check if early stopping is supported
            # Check that eval_set is supported
            # TODO: fix sample_weights dropping & evaluations of the early stopping
        model.fit(  # type: ignore
            X_train=np.asarray(X),
            y_train=y,)

        self.is_fitted = True
    def _predict_quantiles(
            self,
            X: pd.DataFrame,
            quantile: float
            )-> pd.DataFrame:
        q_50, q_quantile = self.model.predcit(X=X, confidence_level = quantile, ensemble=True)
        q_quantile: np.ndarray = q_quantile.reshape(-1,2)
        df = pd.DataFrame(
            data=q_quantile,
            index=X.index,
            columns=[Quantile(quantile).format(), Quantile(1-quantile).format()],
        )
        df[Quantile(0.5).format()] = q_50
        return df.sort_index(axis=1)

    def predict(self, X: npt.NDArray[np.floating] | pd.DataFrame) -> npt.NDArray[np.floating]:
        """Predict quantiles for the input features.

        Args:
            X: Input features as a DataFrame.

        Returns:

            A 2D array where each column corresponds to predicted quantiles.
        """  # noqa: D412
        # predictions = [model.predict(X=X), alpha = a, ]
        # TODO: adapt predict method to the timeseries.
        confidences: self.quantiles = [: len(self.quantiles)//2]
        predictions = {c: self._predict_quantiles(X, c) for c in confidences}
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


class ConformalizedRegressor(BaseEstimator, RegressorMixin):
    """performs multi-quantile regression using a base learner.
    in this conformation, the self._models can be dropped,
    no for loop needed, the training happens inside.

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
        eval_sample_weight: list[npt.NDArray[np.floating]] | list[pd.Series] | None = None,
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
    
        # Check if early stopping is supported
        # Check that eval_set is supported
        if eval_set is None and "early_stopping_rounds" in self.hyperparams:
            model.set_params(early_stopping_rounds=None)  # type: ignore

        if eval_set is not None and self.learner_eval_sample_weight_param is not None:  # type: ignore
            kwargs[self.learner_eval_sample_weight_param] = eval_sample_weight

        if "early_stopping_rounds" in self.hyperparams and self.learner_eval_sample_weight_param is not None:
            model.set_params(early_stopping_rounds=self.hyperparams["early_stopping_rounds"])  # type: ignore

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

    def predict(self, X: npt.NDArray[np.floating] | pd.DataFrame) -> npt.NDArray[np.floating]:
        """Predict quantiles for the input features.

        Args:
            X: Input features as a DataFrame.

        Returns:

            A 2D array where each column corresponds to predicted quantiles.
        """  # noqa: D412

        return np.column_stack([model.predict(X=X) for model in self._models])  # type: ignore


    @property
    def has_feature_names(self) -> bool:
        """Check if the base learners have feature names.

        Returns:
            True if the base learners have feature names, False otherwise.
        """
        return len(self.model_feature_names) > 0


# class ConformalizeQuantileRegressor(BaseEstimator, RegressorMixin):
#     """Adaptor for Conformalized Quantile Regression (CQR) using MapieQuantileRegressor.

#     It wraps a base quantile learner to produce a CQR prediction interval, defined by
#     the minimum and maximum values in the 'quantiles' list.
#     """

#     def __init__(
#         self,
#         base_learner:list[BaseEstimator],
#         quantile_param: str,
#         quantiles: list[float],
#         hyperparams: dict[str, ParamType],
#     ):
#         """Initialize ConformalizeQuantileRegressor.

#         Args:
#             base_learner: A scikit-learn compatible regressor class that supports quantile regression.
#             quantile_param: The name of the parameter in base_learner that sets the quantile level (ignored by Mapie).
#             quantiles: List of quantiles. Only the min/max are used to define the CQR coverage.
#             hyperparams: Dictionary of hyperparameters to pass to the base learner instance.
#         """
#         self.quantiles = sorted(quantiles)
#         self.hyperparams = hyperparams
#         self.quantile_param = quantile_param
#         self.base_learner: list[BaseEstimator] = base_learner
#         self.is_fitted = False
#         self._mapie_model = None

#         # 1. Determine coverage (alpha) for MAPIE from the input quantiles.
#         # The bounds q_low and q_high define the non-conformal interval, and 
#         # the CQR coverage is set to be 1 - (q_high - q_low).
#         # in 1 condition: 
        
        
        # if self.quantiles == [1-x for x in self.quantiles] and len(self.quantiles)%2 ==1 and self.quantiles[int(len(self.quantiles)/2)]==0.5:
        #     raise ValueError(
        #         "Conformalized Quantile Regression requires at least three quantiles and uneven lenght and be symmetrical around 0.5"
        #     )
        # alpha = self.quantiles[:len(self.quantiles)//2]
#         # 2. Initialize base learner and MapieQuantileRegressor.
#         # Mapie handles setting the quantile parameter on the base learner internally (usually to alpha/2 and 1-alpha/2).
#         base_params = self.hyperparams.copy()
#         if self.quantile_param in base_params: 
#             del base_params[self.quantile_param] # Remove quantile_param as Mapie manages it

#         single_base_estimator = self.base_learner(**base_params)

#         # We fix cv="split" as it is the standard CQR strategy that requires calibration data.
#         self._mapie_model = ConformalizedQuantileRegressor(
#             estimator=single_base_estimator,
#             alpha=mapie_alpha,
#             cv="split",
#         )
        
#         # Store index for median if requested, to match output order
#         self.return_median = 0.5 in self.quantiles
#         self.median_index = self.quantiles.index(0.5) if self.return_median else -1

#     def _init_mapie_model():
#         base_learner = self.base_learner(**self.hyperparams)        
        
#         pass

#     def fit(
#         self,
#         X: npt.NDArray[np.floating] | pd.DataFrame,
#         y: npt.NDArray[np.floating] | pd.Series,
#         sample_weight: npt.NDArray[np.floating] | pd.Series | None = None,
#         feature_name: list[str] | None = None,
#         eval_set: list[tuple[pd.DataFrame, npt.NDArray[np.floating]]] | None = None,
#         eval_sample_weight: list[npt.NDArray[np.floating]] | list[pd.Series] | None = None,
#     ) -> None:
#         """Fit the conformalized quantile regressor.

#         This method splits the input data (X, y) into a training set and a
#         calibration set, then fits the MapieQuantileRegressor.
#         If 'eval_set' is provided, its first element is used as the calibration set.
#         """
#         X_data = np.asarray(X)
#         y_data = np.asarray(y)
        
#         X_calib = None
#         y_calib = None
        
#         # If eval_set is provided, use the first set as the calibration data
#         if eval_set is not None and len(eval_set) > 0:
#             X_train = X_data
#             y_train = y_data
#             X_calib = np.asarray(eval_set[0][0])
#             y_calib = np.asarray(eval_set[0][1])
#         else:
#             # Otherwise, perform an internal split of the input data
#             X_train, X_calib, y_train, y_calib = train_test_split(
#                 X_data, y_data, test_size=0.5, random_state=42
#             )

#         # Fit MapieQuantileRegressor using the train and calibration split
#         self._mapie_model.fit(
#             X=X_train,
#             y=y_train,
#             X_calib=X_calib,
#             y_calib=y_calib,
#             sample_weight=sample_weight,
#         )

#         self.is_fitted = True

#     def predict(self, X: npt.NDArray[np.floating] | pd.DataFrame) -> npt.NDArray[np.floating]:
#         """Predict quantiles for the input features.

#         Returns:
#             A 2D array where each column corresponds to predicted quantiles,
#             matching the order of quantiles passed in initialization.
#             Columns that cannot be derived from the CQR interval bounds (e.g., intermediate quantiles)
#             are filled with NaN.
#         """
#         if not self.is_fitted:
#             raise NotFittedError(f"{self.__class__.__name__} is not fitted.")

#         # y_pred_point is the point prediction (often median from the base estimator)
#         # y_pis is the prediction interval: (n_samples, 2, 1) -> (lower bound, upper bound)
#         y_pred_point, y_pis = self._mapie_model.predict(X=X)

#         # Extract lower and upper bounds
#         y_lower = y_pis[:, 0, 0].reshape(-1, 1)
#         y_upper = y_pis[:, 1, 0].reshape(-1, 1)
        
#         n_samples = y_lower.shape[0]
#         output = np.full((n_samples, len(self.quantiles)), np.nan, dtype=np.floating)

#         # Determine indices of the min/max quantiles in the input list
#         q_low = self.quantiles[0]
#         q_high = self.quantiles[-1]
#         q_low_idx = self.quantiles.index(q_low)
#         q_high_idx = self.quantiles.index(q_high)

#         # Place the CQR bounds into the correct columns
#         output[:, q_low_idx] = y_lower.flatten()
#         output[:, q_high_idx] = y_upper.flatten()
        
#         # If the median was requested, place the point prediction there
#         if self.return_median:
#             output[:, self.median_index] = y_pred_point.flatten()

#         return output

#     @property
#     def models(self) -> list[BaseEstimator]:
#         """Get the list of underlying models.

#         Returns:
#             List containing the single MapieQuantileRegressor instance.
#         """
#         return [self._mapie_model] if self._mapie_model else []

#     @property
#     def learner_eval_sample_weight_param(self) -> str | None:
#         """Get the name of the sample weight parameter for evaluation sets.

#         Returns:
#             None, as this complex parameter logic is removed for CQR.
#         """
#         return None
        
#     @property
#     def has_feature_names(self) -> bool:
#         """Check if the base learners have feature names.

#         Returns:
#             False, as feature name handling is removed for simplicity with MAPIE.
#         """
#         return False