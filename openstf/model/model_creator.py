# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union

from openstf_dbc.services.prediction_job import PredictionJobDataClass

from openstf.enums import MLModelType
from openstf.model.regressors.lgbm import LGBMOpenstfRegressor
from openstf.model.regressors.proloaf import OpenstfProloafRegressor
from openstf.model.regressors.regressor import OpenstfRegressor
from openstf.model.regressors.xgb import XGBOpenstfRegressor
from openstf.model.regressors.xgb_quantile import XGBQuantileOpenstfRegressor

# Model parameters needed for init from prediction job
model_init_args = {
    MLModelType.XGB: [],
    MLModelType.LGB: [],
    MLModelType.XGB_QUANTILE: [
        "quantiles",
    ],
    MLModelType.ProLoaf: [
        "horizon_minutes",
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
    }

    @staticmethod
    def create_model(
        pj: Union[PredictionJobDataClass, dict],
    ) -> OpenstfRegressor:
        """Create a machine learning model based on model type.

        Args:
            pj (Union[PredictionJobDataClass, dict]): prediction job

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
                "No constructor for '{}', valid model_types are: {}".format(
                    pj["model"], valid_types
                )
            ) from e

        # only pass relevant arguments to model constructor to prevent warnings
        model_kwargs = {key: pj[key] for key in model_init_args[model_type]}
        return ModelCreator.MODEL_CONSTRUCTORS[model_type](**model_kwargs)
