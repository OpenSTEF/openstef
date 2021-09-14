# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
from xgboost import XGBRegressor
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface


class XGBOpenstfRegressor(XGBRegressor, OpenstfRegressorInterface):
    """XGB Regressor which implements the Openstf regressor API."""

    def get_feature_importance(self):
        self.importance_type = "total_gain"
        gain = self.feature_importances_

        self.importance_type = "weight"
        number = self.feature_importances_
        number = (number / sum(number)) * 100

        feature_importance = pd.DataFrame({"gain": gain, "number": number}, index=self.feature_names)
        return feature_importance
