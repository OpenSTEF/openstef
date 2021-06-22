# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.utils import TestData
from test.utils import BaseTestCase

from openstf.pipeline.train_predict_backtest import train_model_and_forecast_back_test


class TestTrainBackTestPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        self.train_input = TestData.load("reference_sets/307-train-data.csv").head(2000)

    def test_train_model_pipeline_core_happy_flow(self):
        """Test happy flow of the train model pipeline

        NOTE this does not explain WHY this is the case?
        The input data should not contain features (e.g. T-7d),
        but it can/should include predictors (e.g. weather data)

        """

        forecast = train_model_and_forecast_back_test(
            pj=self.pj,
            input_data=self.train_input,
            trained_models_folder="TEST",
            save_figures_folder="OTHER_TEST",
            training_horizons=[0.25, 24.0],
        )

        self.assertTrue("forecast" in forecast.columns.to_list())
        self.assertTrue("realised" in forecast.columns.to_list())
        self.assertTrue("Horizon" in forecast.columns.to_list())
        self.assertEqual(list(forecast.Horizon.unique()), [24.0, 0.25])
