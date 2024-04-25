# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.base
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import Booster

import openstef.metrics.metrics as metrics
from openstef.model.regressors.regressor import OpenstfRegressor

DEFAULT_QUANTILES: tuple[float, ...] = (0.9, 0.5, 0.1)


class XGBMultiOutputQuantileOpenstfRegressor(OpenstfRegressor):
    r"""Model that provides multioutput quantile regression with XGBoost by default using the arctan loss function.

    Arctan loss:
        Refence: https://github.com/LaurensSluyterman/XGBoost_quantile_regression/tree/master
        The key idea is to use a smooth approximation of the pinball loss, the arctan
        pinball loss, that has a relatively large second derivative.

        The approximation is given by:
        $$L^{(\text{arctan})}_{\tau, s}(u) = (\tau - 0.5 + \frac{\arctan (u/s)}{\pi})u   + \frac{s}{\pi}$$.  # noqa E501

        Some important settings:

        * The parameter in the loss function determines the amount of smoothing. A
            smaller values gives a closer approximation but also a much smaller second
            derivative. A larger value gives more conservative quantiles when
            is larger than 0.5, the quantile becomes larger and vice versa.
            Values between 0.05 and 0.1 appear to work well. It may be a good idea to
            optimize this parameter.
        * Set min-child-weight to zero. The second derivatives can be a lot smaller
            than 1 and this parameter may prevent any splits.
        * Use a relatively small max-delta-step. We used a default of 0.5.
            This prevents excessive steps that could happen due to the relatively
            small second derivative.
        * For the same reason, use a slightly lower learning rate of 0.05.

    """

    estimator_: TransformedTargetRegressor
    quantile_indices_: Dict[float, int]

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        gamma: float = 0.0,
        colsample_bytree: float = 1.0,
        subsample: float = 1.0,
        min_child_weight: int = 0,
        max_depth: int = 6,
        learning_rate: float = 0.22,
        alpha: float = 0.0,
        max_delta_step: int = 0.5,
        arctan_smoothing: float = 0.055,
        early_stopping_rounds: Optional[int] = None,
    ):
        """Initialize XGBMultiQuantileRegressor.

        Model that provides quantile regression with XGBoost.
        For each desired quantile an XGBoost model is trained,
        these can later be used to predict quantiles.

        Args:
            quantiles: Tuple with desired quantiles, quantile 0.5 is required.
                For example: (0.1, 0.5, 0.9)
            gamma: Gamma.
            colsample_bytree: Colsample by tree.
            subsample: Subsample.
            min_child_weight: Minimum child weight.
            max_depth: Maximum depth.
            learning_rate: Learning rate.
            alpha: Alpha.
            max_delta_step: Maximum delta step.
            arctan_smoothing: smoothing parameter of the arctan loss function.
            early_stopping_rounds: Number of rounds to stop training if no improvement
                is made.

        Raises:
            ValueError in case quantile 0.5 is not in the requested quantiles.

        """
        super().__init__()
        if 0.5 not in quantiles:
            raise ValueError(
                "Cannot train quantile model as 0.5 is not in requested quantiles!"
            )

        self.quantiles = quantiles

        # Set attributes for hyper parameters
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.gamma = gamma
        self.alpha = alpha
        self.max_delta_step = max_delta_step
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.arctan_smoothing = arctan_smoothing

        # Get fitting parameters - only those required for xgbooster's
        xgb_regressor_params = {
            key: value
            for key, value in self.get_params().items()
            if key in xgb.XGBRegressor().get_params().keys()
        }

        # Define the model
        objective = partial(
            metrics.arctan_loss, taus=self.quantiles, s=arctan_smoothing
        )
        xgb_model: xgb.XGBRegressor = xgb.XGBRegressor(
            objective=objective,
            base_score=0,
            multi_strategy="one_output_per_tree",
            **xgb_regressor_params,
        )
        self.estimator_ = TransformedTargetRegressor(
            regressor=xgb_model, transformer=StandardScaler()
        )

        # Set quantile indices to remap multioutput predictions
        self.quantile_indices_ = {
            quantile: i for i, quantile in enumerate(self.quantiles)
        }

    def fit(
        self,
        x: np.array,
        y: np.array,
        eval_set: Optional[Sequence[Tuple[np.array, np.array]]] = None,
        verbose: Optional[Union[bool, int]] = 0,
        **kwargs
    ) -> OpenstfRegressor:
        """Fits xgb quantile model.

        Args:
            x: Feature matrix.
            y: Labels.
            eval_set: Evaluation set to monitor training performance.
            verbose: Verbosity level (disabled by default).

        Returns:
            Fitted XGBQuantile model.

        """
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.asarray(x))

        # Check/validate input
        check_X_y(x, y, force_all_finite="allow-nan")

        # Prepare inputs
        y_multioutput = replicate_for_multioutput(y, len(self.quantiles))

        # Define watchlist if eval_set is defined
        eval_set_multioutput = []
        if eval_set:
            for x_eval, y_eval in eval_set:
                if isinstance(y_eval, pd.Series):
                    y_eval = y_eval.to_numpy()

                y_eval_multioutput = replicate_for_multioutput(
                    y=y_eval, num_quantiles=len(self.quantiles)
                )
                eval_set_multioutput.append((x_eval, y_eval_multioutput))

            eval_set_multioutput.append((x, y_multioutput))

        self.estimator_.fit(
            X=x.copy(deep=True),
            y=y_multioutput,
            eval_set=eval_set_multioutput,
            verbose=verbose,
        )

        # Update state of the estimator
        self.feature_importances_ = self.estimator_.regressor_.feature_importances_
        self.is_fitted_ = True

        return self

    def predict(self, x: np.array, quantile: float = 0.5) -> np.array:
        """Makes a prediction for a desired quantile.

        Args:
            x: Feature matrix.
            quantile: Quantile for which a prediciton is desired,
                note that only quantile are available for which a model is trained,
                and that this is a quantile-model specific keyword.

        Returns:
            Prediction

        Raises:
            ValueError in case no model is trained for the requested quantile.

        """
        # Check if model is trained for this quantile
        if quantile not in self.quantiles:
            raise ValueError("No model trained for requested quantile!")

        # Check/validate input
        check_array(x, force_all_finite="allow-nan")
        check_is_fitted(self)

        # best_iteration is only available if early stopping was used during training
        prediction: np.array
        if hasattr(self.estimator_, "best_iteration"):
            prediction = self.estimator_.predict(
                X=x,
                iteration_range=(0, self.estimator_.best_iteration + 1),
            )
        else:
            prediction = self.estimator_.predict(X=x)

        quantile_index = self.quantile_indices_[quantile]
        return prediction[:, quantile_index]

    @property
    def feature_names(self):
        return self.estimator_.feature_names_in_

    @property
    def can_predict_quantiles(self):
        return True


def replicate_for_multioutput(y: np.array, num_quantiles: int) -> np.array:
    """Replicates a 1D array to a 2D array for multioutput regression.

    Args:
        y: 1D array.
        num_quantiles: Number of columns in the output array.

    Returns:
        2D array with shape (len(y), num_quantiles)

    """
    return np.repeat(y[:, None], num_quantiles, axis=1)
