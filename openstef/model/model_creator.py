# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import logging
from typing import Union

import structlog

from openstef.enums import ModelType
from openstef.model.regressors.arima import ARIMAOpenstfRegressor
from openstef.model.regressors.custom_regressor import is_custom_type, load_custom_model
from openstef.model.regressors.gblinear_quantile import GBLinearQuantileOpenstfRegressor
from openstef.model.regressors.lgbm import LGBMOpenstfRegressor
from openstef.model.regressors.linear import LinearOpenstfRegressor
from openstef.model.regressors.linear_quantile import LinearQuantileOpenstfRegressor
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.regressors.flatliner import FlatlinerRegressor
from openstef.model.regressors.xgb import XGBOpenstfRegressor
from openstef.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor
from openstef.model.regressors.xgb_multioutput_quantile import (
    XGBMultiOutputQuantileOpenstfRegressor,
)
from openstef.settings import Settings

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(Settings.log_level)
    )
)
logger = structlog.get_logger(__name__)

valid_model_kwargs = {
    ModelType.XGB: [
        "n_estimators",
        "objective",
        "max_depth",
        "learning_rate",
        "verbosity",
        "booster",
        "tree_method",
        "gamma",
        "min_child_weight",
        "max_delta_step",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "colsample_bynode",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "base_score",
        "missing",
        "num_parallel_tree",
        "kwargs",
        "random_state",
        "n_jobs",
        "monotone_constraints",
        "interaction_constraints",
        "importance_type",
        "gpu_id",
        "validate_parameters",
        "early_stopping_rounds",
    ],
    ModelType.LGB: [
        "boosting_type",
        "objective",
        "num_leaves",
        "max_depth",
        "learning_rate",
        "n_estimators",
        "subsample_for_bin",
        "min_split_gain",
        "min_child_weight",
        "min_child_samples",
        "subsample",
        "subsample_freq",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "random_state",
        "n_jobs",
        "silent",
        "importance_type",
        "early_stopping_rounds",
    ],
    ModelType.XGB_QUANTILE: [
        "quantiles",
        "gamma",
        "colsample_bytree",
        "subsample",
        "min_child_weight",
        "max_depth",
        "early_stopping_rounds",
    ],
    ModelType.XGB_MULTIOUTPUT_QUANTILE: [
        "quantiles",
        "gamma",
        "colsample_bytree",
        "subsample",
        "min_child_weight",
        "max_depth",
        "early_stopping_rounds",
        "arctan_smoothing",
    ],
    ModelType.LINEAR: [
        "missing_values",
        "imputation_strategy",
        "fill_value",
    ],
    ModelType.FLATLINER: [
        "quantiles",
    ],
    ModelType.LINEAR_QUANTILE: [
        "alpha",
        "quantiles",
        "solver",
        "missing_values",
        "imputation_strategy",
        "fill_value",
        "weight_scale_percentile",
        "weight_exponent",
        "weight_floor",
        "no_fill_future_values_features",
    ],
    ModelType.GBLINEAR_QUANTILE: [
        "quantiles",
        "missing_values",
        "imputation_strategy",
        "fill_value",
        "weight_scale_percentile",
        "weight_exponent",
        "weight_floor",
        "no_fill_future_values_features",
        "clipped_features",
        "learning_rate",
        "num_boost_round",
        "early_stopping_rounds",
        "reg_alpha",
        "reg_lambda",
        "updater",
        "feature_selector",
        "top_k",
    ],
    ModelType.ARIMA: [
        "backtest_max_horizon",
        "order",
        "seasonal_order",
        "trend",
    ],
}


class ModelCreator:
    """Factory object for creating machine learning models."""

    # Set object mapping
    MODEL_CONSTRUCTORS = {
        ModelType.XGB: XGBOpenstfRegressor,
        ModelType.LGB: LGBMOpenstfRegressor,
        ModelType.XGB_QUANTILE: XGBQuantileOpenstfRegressor,
        ModelType.XGB_MULTIOUTPUT_QUANTILE: XGBMultiOutputQuantileOpenstfRegressor,
        ModelType.LINEAR: LinearOpenstfRegressor,
        ModelType.LINEAR_QUANTILE: LinearQuantileOpenstfRegressor,
        ModelType.GBLINEAR_QUANTILE: GBLinearQuantileOpenstfRegressor,
        ModelType.ARIMA: ARIMAOpenstfRegressor,
        ModelType.FLATLINER: FlatlinerRegressor,
    }

    @staticmethod
    def create_model(model_type: Union[ModelType, str], **kwargs) -> OpenstfRegressor:
        """Create a machine learning model based on model type.

        Args:
            model_type: Model type to construct.
            kwargs: Optional keyword argument to pass to the model.

        Raises:
            NotImplementedError: When using an invalid model_type.

        Returns:
            OpenSTEF model

        """
        try:
            # This will raise a ValueError when an invalid model_type str is used
            # and nothing when a MLModelType enum is used.
            if is_custom_type(model_type):
                model_class = load_custom_model(model_type)
                valid_kwargs = model_class.valid_kwargs()
            else:
                model_type = ModelType(model_type)
                model_class = ModelCreator.MODEL_CONSTRUCTORS[model_type]
                valid_kwargs = valid_model_kwargs[model_type]
                # Check if model as imported
                if model_class is None:
                    raise ImportError(
                        f"Constructor not available for '{model_type}'. "
                        "Perhaps you forgot to install an optional dependency? "
                        "Please refer to the ReadMe for instructions"
                    )
        except ValueError as e:
            valid_types = [t.value for t in ModelType]
            raise NotImplementedError(
                f"No constructor for '{model_type}', "
                f"valid model_types are: {valid_types} "
                "or import a custom model"
            ) from e

        # only pass relevant arguments to model constructor to prevent warnings
        model_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_kwargs
        }

        return model_class(**model_kwargs)
