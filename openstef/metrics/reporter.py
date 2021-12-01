# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import sklearn
from mlflow.models import ModelSignature, infer_signature
from plotly.graph_objects import Figure

from openstef.metrics import figure
from openstef.metrics.metrics import bias, mae, nsme, r_mae, rmse
from openstef.model.regressors.regressor import OpenstfRegressor


@dataclass
class Report:
    def __init__(
        self,
        feature_importance_figure: Figure,
        data_series_figures: Dict[str, Figure],
        metrics: dict,
        signature: ModelSignature,
    ):
        self.feature_importance_figure = feature_importance_figure
        self.data_series_figures = data_series_figures
        self.metrics = metrics
        self.signature = signature


class Reporter:
    def __init__(
        self,
        train_data: pd.DataFrame = None,
        validation_data: pd.DataFrame = None,
        test_data: pd.DataFrame = None,
    ) -> None:
        """Initializes reporter

        Args:
            train_data: Dataframe with training data
            validation_data: Dataframe with validation data
            test_data: Dataframe with test data
        """
        self.horizons = train_data.horizon.unique()
        self.predicted_data_list = []
        self.input_data_list = [train_data, validation_data, test_data]

    def generate_report(
        self,
        model: OpenstfRegressor,
    ) -> Report:
        """Generate a report on a given model

        Args:
            model (OpenstfRegressor): the model to create a report on

        Returns:
            Report: reporter object containing info about the model
        """
        # Get training (input_data_list[0]) and validation (input_data_list[1]) set
        train_x, train_y = (
            self.input_data_list[0].iloc[:, 1:-1],
            self.input_data_list[0].iloc[:, 0],
        )
        valid_x, valid_y = (
            self.input_data_list[1].iloc[:, 1:-1],
            self.input_data_list[1].iloc[:, 0],
        )

        data_series_figures = self._make_data_series_figures(model)

        # feature_importance_dataframe should be a dataframe, to create a figure
        # can be None if we have no feature importance
        if isinstance(model.feature_importance_dataframe, pd.DataFrame):
            feature_importance_figure = figure.plot_feature_importance(
                model.feature_importance_dataframe
            )
        # If it isn't a dataframe we will set feature_importance_figure, so it will not create the figure
        else:
            feature_importance_figure = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = Report(
                data_series_figures=data_series_figures,
                feature_importance_figure=feature_importance_figure,
                metrics=self.get_metrics(model.predict(valid_x), valid_y),
                signature=infer_signature(train_x, train_y),
            )

        return report

    @staticmethod
    def get_metrics(y_pred: np.array, y_true: np.array) -> dict:
        """Calculate the metrics for a prediction

        Args:
            y_pred: np.array
            y_true: np.array

        Returns:
            dictionary: metrics for the prediction
        """
        metric_dict = {
            "bias": bias,
            "NSME": nsme,
            "MAE": mae,
            "R_MAE": r_mae,
            "RMSE": rmse,
            "explained_variance": sklearn.metrics.explained_variance_score,
            "MSE": sklearn.metrics.mean_squared_error,
            "r2": sklearn.metrics.r2_score,
        }
        results = {}
        for name, metric in metric_dict.items():
            try:
                results[name] = metric(y_true, y_pred)
            except ValueError:
                continue
        return results

    @staticmethod
    def write_report_to_disk(report: Report, location: Path):
        """Write report to disk,
        easy for e.g. viewing report of latest models using grafana"""
        # create path if does not exist
        if not os.path.exists(location):
            os.makedirs(location)
        # write feature importance figure
        report.feature_importance_figure.write_html(f"{location}/weight_plot.html")
        # write predictors
        for name, figure in report.data_series_figures.items():
            figure.write_html(f"{location}/{name}.html")

    def _make_data_series_figures(self, model: OpenstfRegressor) -> dict:
        # Make model predictions
        for data_set in self.input_data_list:
            # First ("load") and last ("horizon") are removed here
            # as they are not expected by the model as prediction input
            model_forecast = model.predict(data_set.iloc[:, 1:-1])
            forecast = pd.DataFrame(
                index=data_set.index, data={"forecast": model_forecast}
            )
            self.predicted_data_list.append(forecast)

        # Make cufflinks plots for the data series
        return {
            f"Predictor{horizon}": figure.plot_data_series(
                data=self.input_data_list,
                predict_data=self.predicted_data_list,
                horizon=horizon,
            )
            for horizon in self.horizons
        }
