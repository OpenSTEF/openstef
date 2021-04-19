# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from openstf.enums import MLModelType
from openstf.model.serializer.xgboost.quantile import XGBQuantileModelSerializer
from openstf.model.serializer.xgboost.xgboost import XGBModelSerializer
from openstf.model.serializer.lightgbm.lightgbm import LGBModelSerializer


class ModelSerializerCreator:

    MODEL_SERIALIZER_CONTRUCTORS = {
        MLModelType.XGB: XGBModelSerializer,
        MLModelType.XGB_QUANTILE: XGBQuantileModelSerializer,
        MLModelType.LGB: LGBModelSerializer,
    }

    @classmethod
    def create_model_serializer(cls, model_type):

        if model_type not in cls.MODEL_SERIALIZER_CONTRUCTORS:
            raise KeyError(f"Unkown model_type: '{model_type}'")

        return cls.MODEL_SERIALIZER_CONTRUCTORS[model_type]()
