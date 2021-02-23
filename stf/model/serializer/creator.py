# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from stf.model.general import MLModelType
from stf.model.serializer.xgboost.quantile import XGBQuantileModelSerializer
from stf.model.serializer.xgboost.xgboost import XGBModelSerializer

# from stf.model.serializer.lightgbm.quantile import LGBQuantileModelSerializer
from stf.model.serializer.lightgbm.lightgbm import LGBModelSerializer


class ModelSerializerCreator:

    MODEL_SERIALIZER_CONTRUCTORS = {
        MLModelType.XGB: XGBModelSerializer,
        MLModelType.XGB_QUANTILE: XGBQuantileModelSerializer,
        MLModelType.LGB: LGBModelSerializer,
        # MLModelType.LGB_QUANTILE: LGBQuantileModelSerializer
    }

    @classmethod
    def create_model_serializer(cls, model_type):

        if model_type not in cls.MODEL_SERIALIZER_CONTRUCTORS:
            raise KeyError(f"Unkown model_type: '{model_type}'")

        return cls.MODEL_SERIALIZER_CONTRUCTORS[model_type]()
