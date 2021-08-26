# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from xgboost import XGBRegressor
from openstf.model.regressors.abstract_stf_regressor import AbstractStfRegressor


class XGBStfRegressor(XGBRegressor, AbstractStfRegressor):
    """Inherits all/most functionality directly from XGBRegressor
    If the AbstractStfRegressor interface changes, we potentially need to include
    functionality here.
    """

    pass
