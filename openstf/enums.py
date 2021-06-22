# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from enum import Enum, auto


# TODO replace this with ModelType (MLModelType == Machine Learning model type)
class MLModelType(Enum):
    XGB = "xgb"
    XGB_QUANTILE = "xgb_quantile"
    LGB = "lgb"


class ForecastType(Enum):
    DEMAND = "demand"
    WIND = "wind"
    SOLAR = "solar"
    BASECASE = "basecase"


class TracyJobResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"
