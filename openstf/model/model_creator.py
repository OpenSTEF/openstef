# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union

from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from openstf.enums import MLModelType


class ModelCreator:
    """Factory object for creating machine learning models"""

    # Set object mapping
    MODEL_CONSTRUCTORS = {
        MLModelType.XGB: XGBRegressor,
        MLModelType.LGB: LGBMRegressor,
    }

    @staticmethod
    def create_model(model_type: Union[MLModelType, str]) -> RegressorMixin:
        """Create a machine learning model based on model type.

        Args:
            model_type (Union[MLModelType, str]): Model type

        Raises:
            ValueError: When using an invalid model_type string

        Returns:
            RegressorMixin: model
        """
        # This will raise a ValueError when an invalid model_type str is used
        model_type = MLModelType(model_type)

        return ModelCreator.MODEL_CONSTRUCTORS[model_type]()
