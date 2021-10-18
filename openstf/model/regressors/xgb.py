# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from xgboost import XGBRegressor

from openstf.model.regressors.regressor import OpenstfRegressor


class XGBOpenstfRegressor(XGBRegressor, OpenstfRegressor):
    """XGB Regressor which implements the Openstf regressor API."""

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }
