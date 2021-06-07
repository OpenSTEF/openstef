# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import scipy
import xgboost as xgb
import structlog

from openstf.validation import validation
from openstf.model.prediction.prediction import AbstractPredictionModel


class XGBPredictionModel(AbstractPredictionModel):
    def __init__(
        self, pj, forecast_type, trained_model=None, trained_confidence_df=None
    ):
        super().__init__(pj, forecast_type, trained_model, trained_confidence_df)
        self.logger = structlog.get_logger(self.__class__.__name__)

    @property
    def feature_names(self):
        return self.model.feature_names

    def calculate_completeness(self, forcast_input_data):
        """Calculate the completeness

        Args:
            forcast_input_data (pandas.DataFrame): Input data

        Returns:
            float: The completeness of the input data
        """

        # Check if forcast_input_data is 'complete' enough.
        scores = self.model.get_score(importance_type="gain")

        for feature in self.model.feature_names:
            if feature not in scores.keys():
                scores.update({feature: 0})

        weights = pd.DataFrame(index=[0], data=scores)
        weights = weights.loc[:, list(forcast_input_data.columns)]

        completeness = validation.calc_completeness(
            forcast_input_data,
            weights.values[0],
            time_delayed=True,
        )

        self.logger.info(f"Completeness: {completeness:.2f}")

        return completeness

    def make_forecast(self, forcast_input_data):
        """Make a forecast using a trained XGBoost model

        Args:
            forcast_input_data (pandas.DataFrame): Forcast (section) input data

        Returns:
            pandas.DataFrame: The forecase
        """
        self.logger.info("Use XGBoost model to make a forecast")

        forecast_request = xgb.DMatrix(forcast_input_data)

        model_forecast = self.model.predict(
            forecast_request, ntree_limit=self.model.best_ntree_limit
        )
        forecast = pd.DataFrame(
            index=forcast_input_data.index, data={"forecast": model_forecast}
        )

        # Add standard deviation from `confidence_df` or file (if exists and is not empty)
        forecast = self._add_standard_deviation_to_forecast(forecast)

        forecast = self._add_quantiles_to_forecast(forecast, self.pj["quantiles"])

        # Add prediction job property to forecast
        forecast = self.add_prediction_job_properties_to_forecast(
            pj=self.pj,
            forecast=forecast,
            algorithm_type=str(self.model_file_path),
            forecast_type=self.forecast_type,
        )

        npoints = len(forecast)

        self.logger.info(f"Made a forecast, npoints: {npoints}", npoints=npoints)

        return forecast

    @staticmethod
    def _add_quantiles_to_forecast(forecast, quantiles):
        """Add quantiles to forecast.

        Use the standard deviation to calculate the quantiles.

        Args:
            forecast (pd.DataFrame): Forecast (should contain a 'forecast' + 'stdev' column)
            quantiles (list): List with desired quantiles

        Returns:
            (pd.DataFrame): Forecast DataFrame with quantile (e.g. 'quantile_PXX')
                columns added.

        """

        # Check if stdev and forecast are in the dataframe
        if not all(elem in forecast.columns for elem in ["forecast", "stdev"]):
            raise ValueError("Forecast should contain a 'forecast' and 'stdev' column")

        for quantile in quantiles:
            quantile_key = f"quantile_P{quantile * 100:02.0f}"
            forecast[quantile_key] = (
                forecast["forecast"]
                + scipy.stats.norm.ppf(quantile) * forecast["stdev"]
            )

        return forecast
