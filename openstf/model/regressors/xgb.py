# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
from xgboost import XGBRegressor
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface


class XGBOpenstfRegressor(XGBRegressor, OpenstfRegressorInterface):
    """XGB Regressor which implements the Openstf regressor API."""

    def get_feature_importance(self, cols):
        self.importance_type = "total_gain"
        gain = self.feature_importances_
        gain = gain / sum(gain)

        self.importance_type = "weight"
        number = self.feature_importances_
        number = number / sum(number)

        feature_importance = pd.DataFrame({"gain": gain, "weight": number}, index=cols)
        feature_importance.sort_values(by="gain", ascending=False, inplace=True)
        return feature_importance
