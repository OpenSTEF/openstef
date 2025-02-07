# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import glob
import os
import unittest
from test.unit.utils.data import TestData

from openstef.pipeline.create_forecast import create_forecast_pipeline_core
from openstef.pipeline.optimize_hyperparameters import optimize_hyperparameters_pipeline
from openstef.pipeline.train_model import train_pipeline_common


class TestComponent(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.input_data = TestData.load("reference_sets/307-train-data.csv")
        self.forecast_data = TestData.load("reference_sets/307-test-data.csv")
        self.pj, self.model_specs = TestData.get_prediction_job_and_modelspecs(pid=307)

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
        parameters = optimize_hyperparameters_pipeline(
            self.pj,
            self.input_data,
            mlflow_tracking_uri="./test/component/mlruns",
            artifact_folder="./test/component",
            n_trials=2,
        )
        self.model_specs.hyper_params = parameters

        # Assert that reports on training are stored in correct location
        expected_report_location = f'./test/component/{self.pj["id"]}'
        fnames = [
            os.path.basename(file_with_path)
            for file_with_path in glob.glob(os.path.join(expected_report_location, "*"))
        ]
        expected_fnames = [
            "Predictor0.25.html",
            "Predictor47.0.html",
            "weight_plot.html",
        ]
        self.assertCountEqual(fnames, expected_fnames)

        # 2) train model, using the optimized hyperparameters
        (
            model,
            report,
            train_data,
            validation_data,
            test_data,
            operational_score_data,
        ) = train_pipeline_common(
            self.pj, self.model_specs, self.input_data, [0.25, 47.0]
        )

        # 3) create forecast, using the trained model
        forecast_data = self.forecast_data
        col_name = forecast_data.columns[0]
        forecast_data.loc["2020-11-28 00:00:00":"2020-12-01", col_name] = None

        run_id_model = os.listdir("./test/component/mlruns/0/")[0]
        model.path = f"./test/component/mlruns/0/{run_id_model}/artifacts/model/"

        forecast = create_forecast_pipeline_core(
            self.pj, forecast_data, model, self.model_specs
        )
        forecast["realised"] = forecast_data.iloc[:, 0]
        forecast["horizon"] = forecast_data.iloc[:, -1]

        # Verify forecast works correctly
        self.assertIn("forecast", forecast.columns)
        self.assertIn("realised", forecast.columns)
        self.assertIn("horizon", forecast.columns)
