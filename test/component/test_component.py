# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from test.unit.utils.data import TestData
import os
from openstef.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline
from openstef.pipeline.train_model import train_pipeline_common
from openstef.pipeline.create_forecast import create_forecast_pipeline_core


class TestComponent(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.input_data = TestData.load("reference_sets/307-train-data.csv")
        self.forecast_data = TestData.load("reference_sets/307-test-data.csv")
        self.pj, self.modelspecs = TestData.get_prediction_job_and_modelspecs(pid=307)

    def test_component_training_prediction_happyflow(self):
        """
        Performs the component test for the pipeline
        1) optimize hyperparameters
        2) train model
        3) make prognoses
        """

        # 1) optimize hyperparameters
        predefined_quantiles = (0.001, 0.5)
        self.pj["quantiles"] = predefined_quantiles
        try:
            parameters = optimize_hyperparameters_pipeline(
                self.pj, self.input_data, "./trained_models", n_trials=2
            )
            self.modelspecs.hyper_params = parameters
        except:
            print("Optimization of hyperparameters failed during the component test")

        # 2) train model, using the optimized hyperparameters
        try:
            (
                model,
                report,
                train_data,
                validation_data,
                test_data,
            ) = train_pipeline_common(
                self.pj, self.modelspecs, self.input_data, [0.25, 47.0]
            )
        except:
            print("Training of the model failed during the component test")

        # 3) create forecast, using the trained model
        forecast_data = self.forecast_data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None

        try:
            run_id_model = next(os.walk("./trained_models/mlruns/1/"))[1][0]
            model.path = f"./trained_models/mlruns/1/{run_id_model}/artifacts/model/"
        except:
            print("Trained model could not be found in the trained_models folder")

        try:
            forecast = create_forecast_pipeline_core(
                self.pj, forecast_data, model, self.modelspecs
            )
            forecast["realised"] = forecast_data.iloc[:, 0]
            forecast["horizon"] = forecast_data.iloc[:, -1]
        except:
            print("Creating a forecast failed during the component test")

        # Verify forecast works correctly
        self.assertTrue("forecast" in forecast.columns)
        self.assertTrue("realised" in forecast.columns)
        self.assertTrue("horizon" in forecast.columns)


if __name__ == "__main__":
    unittest.main()
