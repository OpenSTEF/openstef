# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from openstf.enums import MLModelType
from openstf.model.xgb_quantile import XGBQuantileRegressor


class ModelCreator:

    """Factory object for creating model trainer objects"""

    # Set object mapping
    MODEL_TRAINER_CONSTRUCTORS = {
        MLModelType.XGB: XGBRegressor,
        MLModelType.LGB: LGBMRegressor,
        MLModelType.XGB_QUANTILE: XGBQuantileRegressor,
    }

    @staticmethod
    def create_model(pj: dict) -> RegressorMixin:
        # check if model type is valid
        if pj["model"] not in [
            k.value for k in ModelCreator.MODEL_TRAINER_CONSTRUCTORS
        ]:
            raise KeyError(f"Unknown model type: {pj['model']}")

        return ModelCreator.MODEL_TRAINER_CONSTRUCTORS[MLModelType(pj["model"])](
            quantiles=tuple(pj["quantiles"])
        )
