# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import structlog
import unittest

logger = structlog.get_logger(__name__)

from openstf.model.regressors.xgb import XGBOpenstfRegressor

try:
    from test.unit.utils.base import BaseTestCase
    from openstf.model.regressors.proloaf import OpenstfProloafRegressor
except ImportError:
    logger.warning(
        "Proloaf not available, switching to xgboost! See Readme how to install proloaf dependencies"
    )
    OpenstfProloafRegressor = XGBOpenstfRegressor


class TestProloaf(BaseTestCase):
    def setUp(self) -> None:
        self.model = OpenstfProloafRegressor()

    @unittest.skip  # Skip as this cannot always succeed due to neural network libraries being optional
    def test_importance_names(self):
        self.assertEqual(self.model._get_importance_names(), None)
