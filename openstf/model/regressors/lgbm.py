# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
from lightgbm import LGBMRegressor
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface


class LGBMOpenstfRegressor(LGBMRegressor, OpenstfRegressorInterface):
    """LGBM Regressor which implements the Openstf regressor API."""

    def get_feature_importance(self, cols):
        self.importance_type = "gain"
        gain = self.feature_importances_

        self.importance_type = "split"
        number = self.feature_importances_
        number = (number / sum(number)) * 100
        feature_importance = pd.DataFrame({"gain":gain,"number":number}, index=cols)
        return feature_importance