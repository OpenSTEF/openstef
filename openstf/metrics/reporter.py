# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import os
from pathlib import Path

import pandas as pd
from ktpbase.config.config import ConfigManager
from sklearn.base import RegressorMixin

from openstf.metrics import figure


class Reporter:
    def __init__(
        self,
        pj: dict = None,
        train_data: pd.DataFrame = None,
        validation_data: pd.DataFrame = None,
        test_data: pd.DataFrame = None,
    ) -> None:

        """Initializes reporter object

        Args:
            pj:
            train_data:
            validation_data:
            test_data:
        """
        self.pj = pj
        self.horizons = train_data.Horizon.unique()
        self.predicted_data_list = []
        self.input_data_list = [
            train_data,
            validation_data,
            test_data,
        ]
        self.save_path = (
            Path(ConfigManager.get_instance().paths.webroot) / str(self.pj["id"])
        )  # Path were visuals are saved

    def make_and_save_dashboard_figures(self, model: RegressorMixin) -> None:

        self._make_data_series_figures(model)
        self._make_feature_importance_plot(model)
        self._save_dashboard_figures(self.save_path)

    def _make_data_series_figures(self, model: RegressorMixin) -> None:

        # Make model predictions
        for data_set in self.input_data_list:
            model_forecast = model.predict(data_set.iloc[:, 1:])
            forecast = pd.DataFrame(
                index=data_set.index, data={"forecast": model_forecast}
            )
            self.predicted_data_list.append(forecast)

        # Make cufflinks plots for the data series
        self.figure_series = {
            f"Predictor{horizon}": figure.plot_data_series(
                data=self.input_data_list,
                predict_data=self.predicted_data_list,
                horizon=horizon,
            )
            for horizon in self.horizons
        }

    def _make_feature_importance_plot(self, model: RegressorMixin) -> None:

        feature_importance = self._extract_feature_importance(model)

        # Make feature importance plot
        self.feature_importance_plot = figure.plot_feature_importance(
            feature_importance
        )

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

    def _save_dashboard_figures(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        self.feature_importance_plot.write_html(str(save_path / "weight_plot.html"))
        for key, fig in self.figure_series.items():
            fig.write_html(str(save_path / f"{key}.html"), auto_open=False)
