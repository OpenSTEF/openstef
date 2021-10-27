# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from openstf.pipeline.train_create_forecast_backtest import (
    train_model_and_forecast_back_test,
)
from test.utils import BaseTestCase
from test.utils import TestData


class TestTrainBackTestPipeline(BaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

    def test_train_model_pipeline_core_happy_flow(self):
        """Test happy flow of the train model pipeline"""

        (
            forecast,
            model,
            train_data,
            validation_data,
            test_data,
        ) = train_model_and_forecast_back_test(
            pj=self.pj,
            input_data=self.train_input,
            training_horizons=[0.25, 24.0],
        )

        self.assertTrue("forecast" in forecast.columns)
        self.assertTrue("realised" in forecast.columns)
        self.assertTrue("horizon" in forecast.columns)
        self.assertEqual(list(forecast.horizon.unique()), [0.25, 24.0])
