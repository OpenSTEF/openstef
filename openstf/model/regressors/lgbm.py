# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
from lightgbm import LGBMRegressor
from openstf.model.regressors.regressor import OpenstfRegressor


class LGBMOpenstfRegressor(LGBMRegressor, OpenstfRegressor):
    """LGBM Regressor which implements the Openstf regressor API."""

    gain_importance_name = "gain"
    weight_importance_name = "split"
