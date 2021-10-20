# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import warnings
from dataclasses import dataclass
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from mlflow.models import infer_signature, ModelSignature

from openstf.model.regressors.regressor import OpenstfRegressor
from plotly.graph_objects import Figure
from sklearn import metrics


from openstf.metrics import figure
from openstf.metrics.metrics import bias, nsme, mae, r_mae, rmse


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
        bias_value = bias(y_true, y_pred)
        nsme_value = nsme(y_true, y_pred)
        mae_value = mae(y_true, y_pred)
        r_mae_value = r_mae(y_true, y_pred)
        rmse_value = rmse(y_true, y_pred)
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        return {
            "explained_variance": explained_variance,
            "r2": r2,
            "MAE": mae_value,
            "R_MAE": r_mae_value,
            "MSE": mse,
            "RMSE": rmse_value,
            "bias": bias_value,
            "NSME": nsme_value,
        }

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
