# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from lightgbm import LGBMRegressor
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface


class LGBMOpenstfRegressor(LGBMRegressor, OpenstfRegressorInterface):
    """LGBM Regressor which implements the Openstf regressor API."""

    pass
