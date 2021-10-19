# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union

from openstf.model.regressors.regressor import OpenstfRegressor
from openstf_dbc.services.prediction_job import PredictionJobDataClass
from sklearn.base import RegressorMixin

from openstf.enums import MLModelType
from openstf.model.regressors.lgbm import LGBMOpenstfRegressor
from openstf.model.regressors.xgb import XGBOpenstfRegressor
from openstf.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor

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
}


class ModelCreator:
    """Factory object for creating machine learning models"""

    # Set object mapping
    MODEL_CONSTRUCTORS = {
        MLModelType.XGB: XGBOpenstfRegressor,
        MLModelType.LGB: LGBMOpenstfRegressor,
        MLModelType.XGB_QUANTILE: XGBQuantileOpenstfRegressor,
    }

    @staticmethod
    def create_model(pj: Union[PredictionJobDataClass, dict], **kwargs) -> OpenstfRegressor:
        """Create a machine learning model based on model type.

        Args:
            pj (Union[PredictionJobDataClass, dict]): prediction job
            kwargs (dict): Optional keyword argument to pass to the model.

        Raises:
            NotImplementedError: When using an invalid model_type.

        Returns:
            OpenstfRegressor: model
        """
        try:
            # This will raise a ValueError when an invalid model_type str is used
            # and nothing when a MLModelType enum is used.
            model_type = MLModelType(pj["model"])
        except ValueError as e:
            valid_types = [t.value for t in MLModelType]
            raise NotImplementedError(
                "No constructor for '{}', valid model_types are: {}".format(pj["model"], valid_types)
            ) from e

        model = ModelCreator.MODEL_CONSTRUCTORS[model_type]()
        # only pass relevant arguments to model constructor to prevent warnings
        model_kwargs = {
            key: pj[value]
            for key, value in model.init_parameters().items()
            if key in valid_model_kwargs[model_type]
        }
        model.set_params(**model_kwargs)
        return model
