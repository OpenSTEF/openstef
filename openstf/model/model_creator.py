# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from openstf.enums import MLModelType
from openstf.model.xgb_quantile import XGBQuantileRegressor


class ModelCreator:
    """Factory object for creating machine learning models"""

    # Set object mapping
    MODEL_CONSTRUCTORS = {
        MLModelType.XGB: XGBRegressor,
        MLModelType.LGB: LGBMRegressor,
        MLModelType.XGB_QUANTILE: XGBQuantileRegressor,
    }

    @staticmethod
    def create_model(pj: dict) -> RegressorMixin:
        """Create a machine learning model based on model type.

        Args:
            model_type (Union[MLModelType, str]): Model type

        Raises:
            ValueError: When using an invalid model_type string

        Returns:
            RegressorMixin: model
        """
        # This will raise a ValueError when an invalid model_type str is used
        model_type = MLModelType(pj["model"])

        kwargs = {}

        # If the prediction job contains quantiles pass them to the constructor
        if "quantiles" in pj:
            kwargs["quantiles"] = tuple(pj["quantiles"])

        return ModelCreator.MODEL_CONSTRUCTORS[model_type](**kwargs)
