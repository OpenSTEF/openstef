# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
from xgboost import XGBRegressor
from openstf.model.regressors.regressor import OpenstfRegressor


class XGBOpenstfRegressor(XGBRegressor, OpenstfRegressor):
    """XGB Regressor which implements the Openstf regressor API."""

    gain_importance_name = "total_gain"
    weight_importance_name = "weight"
