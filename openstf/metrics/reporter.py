# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict

from plotly.graph_objects import Figure
import pandas as pd
from sklearn.base import RegressorMixin
import structlog

from openstf.metrics import figure


@dataclass
class Report:
    feature_importance_figure: Figure
    data_series_figures: Dict[str, Figure]
    logger = structlog.get_logger("Report")

    def save_figures(self, save_path):
        save_path = Path(save_path)
        os.makedirs(save_path, exist_ok=True)

        self.feature_importance_figure.write_html(str(save_path / "weight_plot.html"))

        for key, fig in self.data_series_figures.items():
            fig.write_html(str(save_path / f"{key}.html"), auto_open=False)
        self.logger.info(f"Saved figures to {save_path}")


class Reporter:
    def __init__(
        self,
        pj: dict = None,
        train_data: pd.DataFrame = None,
        validation_data: pd.DataFrame = None,
        test_data: pd.DataFrame = None,
    ) -> None:

        """Initializes reporter

        Args:
            pj:
            train_data:
            validation_data:
            test_data:
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
        feature_importance_figure = self._make_feature_importance_figure(model)

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

    def _make_feature_importance_figure(self, model: RegressorMixin) -> None:
        feature_importance = self._extract_feature_importance(model)

        return figure.plot_feature_importance(feature_importance)

    def _extract_feature_importance(self, model):
        """Return feature importances and weights of trained model.

        Returns:
            pandas.DataFrame: A DataFrame describing the feature importances and
            weights of the trained model.

        """
        if model is None:
            return None
        model.importance_type = "gain"
        feature_gain = pd.DataFrame(
            model.feature_importances_,
            index=model._Booster.feature_names,
            columns=["gain"],
        )
        feature_gain /= feature_gain.sum()

        model.importance_type = "weight"
        feature_weight = pd.DataFrame(
            model.feature_importances_,
            index=model._Booster.feature_names,
            columns=["weight"],
        )
        feature_weight /= feature_weight.sum()

        feature_importance = pd.merge(
            feature_gain, feature_weight, left_index=True, right_index=True
        )
        feature_importance.sort_values(by="gain", ascending=False, inplace=True)

        return feature_importance
