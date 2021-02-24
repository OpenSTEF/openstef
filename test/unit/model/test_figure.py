# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd

from openstf.model.figure import plot_data_series, plot_feature_importance

from test.utils import BaseTestCase, TestData


class Teopenstfigure(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.data = TestData.load("input_data.pickle")
        self.train_data = TestData.load("input_data_train.pickle")

    def test_plot_data_series(self):
        self.data["Horizon"] = 47.00
        fig = plot_data_series(
            [self.train_data, self.data],
            [
                self.train_data.rename(columns={"load": "forecast"}),
                self.data.rename(columns={"load": "forecast"}),
            ],
        )
        self.assertTrue(hasattr(fig, "write_html"))

    def test_plot_data_series_predict_data_none(self):
        self.data["Horizon"] = 47.00
        fig = plot_data_series([self.train_data, self.data])
        self.assertTrue(hasattr(fig, "write_html"))

    def test_plot_feature_importtance(self):
        feature_importance = pd.DataFrame(
            data={"gain": [0.25, 0.25, 0.5], "weight": [0.5, 0.2, 0.3]},
            index=["a", "b", "c"],
        )

        fig = plot_feature_importance(feature_importance)
        self.assertTrue(hasattr(fig, "write_html"))
