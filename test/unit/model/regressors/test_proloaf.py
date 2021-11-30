# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.base import BaseTestCase
from openstf.model.regressors.proloaf import OpenstfProloafRegressor


class TestProloaf(BaseTestCase):
    def setUp(self) -> None:
        self.model = OpenstfProloafRegressor()

    def test_importance_names(self):
        self.assertEqual(self.model._get_importance_names(), None)
