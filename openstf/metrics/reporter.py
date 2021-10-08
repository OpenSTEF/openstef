# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from sklearn import metrics
from sklearn.base import RegressorMixin

from openstf.metrics import figure
from openstf.metrics.metrics import bias, nsme, mae, r_mae, rmse
from openstf_dbc.services.prediction_job import PredictionJobDataClass


@dataclass
class Report:
    def __init__(
        self, feature_importance_figure: Figure, data_series_figures: Dict[str, Figure]
    ):
        self.feature_importance_figure = feature_importance_figure
        self.data_series_figures = data_series_figures
        self.metrics = None
        self.signature = None

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
        nsme_value = nsme
        mae_value = mae
        r_mae_value = r_mae
        rmse_value = rmse
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


class Reporter:
    def __init__(
        self,
        pj: Union[dict, PredictionJobDataClass] = None,
        train_data: pd.DataFrame = None,
        validation_data: pd.DataFrame = None,
        test_data: pd.DataFrame = None,
    ) -> None:
        """Initializes reporter

        Args:
            pj: Union[dict, PredictionJobDataClass]
            train_data: pd.DataFrame
            validation_data: pd.DataFrame
            test_data: pd.DataFrame
        """
        self.pj = pj
        self.horizons = train_data.horizon.unique()
        self.predicted_data_list = []
        self.input_data_list = [train_data, validation_data, test_data]

    def generate_report(
        self,
        model: RegressorMixin,
    ) -> Report:
        data_series_figures = self._make_data_series_figures(model)
        feature_importance_figure = figure.plot_feature_importance(
            model.feature_importance_dataframe
        )

        report = Report(
            data_series_figures=data_series_figures,
            feature_importance_figure=feature_importance_figure,
        )

        return report

    def _make_data_series_figures(self, model: RegressorMixin) -> dict:
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
