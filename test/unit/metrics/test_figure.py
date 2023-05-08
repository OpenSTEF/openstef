# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import pandas as pd

from openstef.metrics.figure import plot_data_series, plot_feature_importance


class Teopenstefigure(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.data = TestData.load("input_data.csv").rename(
            columns={"Horizon": "horizon"}
        )
        self.train_data = TestData.load("input_data_train.csv").rename(
            columns={"Horizon": "horizon"}
        )

    def test_plot_data_series(self):
        self.data["horizon"] = 47.00
        fig = plot_data_series(
            [self.train_data, self.data],
            [
                self.train_data.rename(columns={"load": "forecast"}),
                self.data.rename(columns={"load": "forecast"}),
            ],
        )
        self.assertTrue(hasattr(fig, "write_html"))

    def test_plot_data_series_predict_data_none(self):
        self.data["horizon"] = 47.00
        fig = plot_data_series([self.train_data, self.data])
        self.assertTrue(hasattr(fig, "write_html"))

    def test_plot_feature_importtance(self):
        feature_importance = pd.DataFrame(
            data={"gain": [0.25, 0.25, 0.5], "weight": [0.5, 0.2, 0.3]},
            index=["a", "b", "c"],
        )

        fig = plot_feature_importance(feature_importance)
        self.assertTrue(hasattr(fig, "write_html"))
