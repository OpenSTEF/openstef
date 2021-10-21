# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.utils import BaseTestCase
from openstf.model.regressors.lgbm import LGBMOpenstfRegressor


class TestXGB(BaseTestCase):
    def setUp(self) -> None:
        self.model = LGBMOpenstfRegressor()

    def test_importance_names(self):
        self.assertIsInstance(self.model._get_importance_names(), dict)
