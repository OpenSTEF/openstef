# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from enum import Enum


class ModelType(Enum):
    XGB = "xgb"
    XGB_QUANTILE = "xgb_quantile"
    XGB_MULTIOUTPUT_QUANTILE = "xgb_multioutput_quantile"
    LGB = "lgb"
    LINEAR = "linear"
    LINEAR_QUANTILE = "linear_quantile"
    ARIMA = "arima"
    FLATLINER = "flatliner"


class ForecastType(Enum):
    DEMAND = "demand"
    WIND = "wind"
    SOLAR = "solar"
    BASECASE = "basecase"


class PipelineType(Enum):
    FORECAST = "forecast"
    TRAIN = "train"
    HYPER_PARMATERS = "hyper_parameters"
