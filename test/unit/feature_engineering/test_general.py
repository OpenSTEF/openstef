# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from unittest import TestCase

import numpy as np
import pandas as pd

from openstf.feature_engineering.general import enforce_feature_order


class TestGeneral(TestCase):
    def test_enforce_feature_order_with_horizon_columns(self):

        df = pd.DataFrame(
            np.arange(12).reshape(3, 4), columns=["load", "horizon", "A", "E"]
        )

        result = enforce_feature_order(df)

        self.assertEqual(result.columns.to_list(), ["load", "A", "E", "horizon"])

    def test_enforce_feature_order(self):

        df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=["load", "A", "E"])
        result = enforce_feature_order(df)
        self.assertEqual(result.columns.to_list(), ["load", "A", "E"])
