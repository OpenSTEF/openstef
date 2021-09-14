# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface


class OpenstfRegressor(OpenstfRegressorInterface):
    def get_feature_importance_base(self, cols, gain_importance_name, weight_importance_name):
        gain = self._fraction_importance(gain_importance_name)
        weight_importance = self._fraction_importance(weight_importance_name)

        feature_importance = pd.DataFrame({"gain": gain, "weight": weight_importance},
                                          index=cols)
        feature_importance.sort_values(by="gain", ascending=False, inplace=True)
        return feature_importance

    def _fraction_importance(self, importance):
        self.importance_type = importance
        feature_importance = self.feature_importances_
        feature_importance = feature_importance / sum(feature_importance)
        return feature_importance
