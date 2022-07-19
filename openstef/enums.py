# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from enum import Enum

# Specify optional modeltypes
try:
    from openstef.model.regressors.proloaf import OpenstfProloafRegressor

    _has_proloaf = True
except ImportError:
    _has_proloaf = False


# TODO replace this with ModelType (MLModelType == Machine Learning model type)
class MLModelType(Enum):
    XGB = "xgb"
    XGB_QUANTILE = "xgb_quantile"
    LGB = "lgb"
    LINEAR = "linear"
    if _has_proloaf:
        ProLoaf = "proloaf"
    else:
        ProLoaf = None


class ForecastType(Enum):
    DEMAND = "demand"
    WIND = "wind"
    SOLAR = "solar"
    BASECASE = "basecase"


class TracyJobResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    UNKNOWN = "unknown"
