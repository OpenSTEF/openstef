import os
from pathlib import Path

import pandas as pd
from sklearn.base import RegressorMixin

from openstf.metrics import figure


class Reporter:

    def __init__(self, pj: dict = None, train_data: pd.DataFrame = None,
                 validation_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None) -> None:

        """ Initializes reporter object

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

    def make_and_save_dashboard_figures(self, model: RegressorMixin, save_path: Path) -> None:
        self._make_data_series_figures(model)
        self._make_feature_importance_plot(model)
        self._save_dashboard_figures(save_path)




    def _make_data_series_figures(self, model: RegressorMixin) -> None:

        # Make model predictions
        for data_set in self.input_data_list:
            self.predicted_data_list.append(model.predict(data_set))

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
        # Make feature importance plot
        self.feature_importance_plot = figure.plot_feature_importance(model.feature_importance)

    def _save_dashboard_figures(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        self.feature_importance_plot.write_html(str(save_path / "weight_plot.html"))
        for key, fig in self.figure_series.items():
            fig.write_html(str(save_path / f"{key}.html"), auto_open=False)
