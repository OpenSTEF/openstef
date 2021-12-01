# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union
import structlog

from openstef.enums import MLModelType
from openstef.model.regressors.lgbm import LGBMOpenstfRegressor
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.regressors.xgb import XGBOpenstfRegressor
from openstef.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor
from openstef.model.regressors.linear import LinearOpenstfRegressor
from openstef.model.regressors.custom_regressor import load_custom_model, is_custom_type

logger = structlog.get_logger(__name__)
try:
    from openstef.model.regressors.proloaf import OpenstfProloafRegressor
except ImportError:
    logger.warning(
        "Proloaf not available, switching to xgboost! See Readme how to install proloaf dependencies"
    )
    OpenstfProloafRegressor = XGBOpenstfRegressor

valid_model_kwargs = {
    MLModelType.XGB: [
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
    ],
    MLModelType.LGB: [
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
    ],
    MLModelType.XGB_QUANTILE: [
        "quantiles",
        "gamma",
        "colsample_bytree",
        "subsample",
        "min_child_weight",
        "max_depth",
    ],
    MLModelType.ProLoaf: [
        "relu_leak",
        "encoder_features",
        "decoder_features",
        "core_layers",
        "rel_linear_hidden_size",
        "rel_core_hidden_size",
        "dropout_fc",
        "dropout_core",
        "training_metric",
        "metric_options",
        "optimizer_name",
        "early_stopping_patience",
        "early_stopping_margin",
        "learning_rate",
        "max_epochs",
        "device",
        "batch_size",
        "history_horizon",
        "horizon_minutes",
    ],
    MLModelType.LINEAR: [
        "missing_values",
        "imputation_strategy",
        "fill_value",
    ],
}


class ModelCreator:
    """Factory object for creating machine learning models"""

    # Set object mapping
    MODEL_CONSTRUCTORS = {
        MLModelType.XGB: XGBOpenstfRegressor,
        MLModelType.LGB: LGBMOpenstfRegressor,
        MLModelType.XGB_QUANTILE: XGBQuantileOpenstfRegressor,
        MLModelType.ProLoaf: OpenstfProloafRegressor,
        MLModelType.LINEAR: LinearOpenstfRegressor,
    }

    @staticmethod
    def create_model(model_type: Union[MLModelType, str], **kwargs) -> OpenstfRegressor:
        """Create a machine learning model based on model type.

        Args:
            model_type (Union[MLModelType, str]): Model type to construct.
            kwargs (dict): Optional keyword argument to pass to the model.

        Raises:
            NotImplementedError: When using an invalid model_type.

        Returns:
            OpenstfRegressor: model
        """
        try:
            # This will raise a ValueError when an invalid model_type str is used
            # and nothing when a MLModelType enum is used.
            if is_custom_type(model_type):
                model_class = load_custom_model(model_type)
                valid_kwargs = model_class.valid_kwargs()
            else:
                model_type = MLModelType(model_type)
                model_class = ModelCreator.MODEL_CONSTRUCTORS[model_type]
                valid_kwargs = valid_model_kwargs[model_type]
        except ValueError as e:
            valid_types = [t.value for t in MLModelType]
            raise NotImplementedError(
                f"No constructor for '{model_type}', "
                f"valid model_types are: {valid_types}"
                f"or import a custom model"
            ) from e

        # only pass relevant arguments to model constructor to prevent warnings
        model_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_kwargs
        }

        return model_class(**model_kwargs)
