# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from openstf.enums import MLModelType


class ModelCreator:
    """Factory object for creating model trainer objects"""

    # Set object mapping
    MODEL_TRAINER_CONSTRUCTORS = {
        MLModelType.XGB: XGBRegressor,
        MLModelType.LGB: LGBMRegressor,
    }

    @staticmethod
    def create_model(model_type):
        # check if model type is valid
        if model_type not in [k.value for k in ModelCreator.MODEL_TRAINER_CONSTRUCTORS]:
            raise KeyError(f"Unknown model type: {model_type}")

        return ModelCreator.MODEL_TRAINER_CONSTRUCTORS[MLModelType(model_type)]()
