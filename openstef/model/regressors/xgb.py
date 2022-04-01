# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from xgboost import XGBRegressor

from openstef.model.regressors.regressor import OpenstfRegressor


class XGBOpenstfRegressor(XGBRegressor, OpenstfRegressor):
    """XGB Regressor which implements the Openstf regressor API."""

    gain_importance_name = "total_gain"
    weight_importance_name = "weight"

    @property
    def feature_names(self):
        return self._Booster.feature_names

    @property
    def can_predict_quantiles(self):
        return False

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }
