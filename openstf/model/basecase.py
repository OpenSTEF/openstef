# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

MINIMAL_RESOLUTION: int = 15  # Used for validating the forecast input


class BaseCaseModel(BaseEstimator, RegressorMixin):
    def predict(self, forecast_input_data: pd.DataFrame) -> pd.DataFrame:
        """Predict using the basecase method. The basecase forecast is determined by the T-7d and T-14d load.
        This means fitting the model is not required.
        However a fit method is still included to be fully comatible with sklearn.

        Args:
            forecast_input_data (pd.DataFrame): Forecast input dataframe

        Returns: pd.DataFrame: Basecase forecast

        """
        return self.make_basecase_forecast(forecast_input_data)

    def fit(self):
        return self

    @staticmethod
    def make_basecase_forecast(
        forecast_input_data: pd.DataFrame, overwrite_delay_hours: int = 48
    ) -> pd.DataFrame:
        """Make a basecase forecast. THe idea of the basecase forecast is that if all else fails, this forecasts is
            still available.
            Basecase example: the load of last week.
        Args:
            forecast_input_data (pd.DataFrame): Forecast input dataframe
            overwrite_delay_hours (float): times before this in the future are not
                forecasted

        Raises:
            ValueError: if columns T-7d or T-14d is not present
            ValueError: If the start of the forecast is before the horizon of the regular forecast
        Returns:
            pd.DataFrame: Basecase forecast


        """
        # Check if required features are provided
        if not all(
            item in forecast_input_data.columns.to_list() for item in ["T-14d", "T-7d"]
        ):
            raise ValueError(
                "Could not make basecase, features T-7d and T-14d are required! Tip: Generate these features with a FeatureApplicator object."
            )

        # Make basecase forecast: Use load of last week
        basecase_forecast = (
            forecast_input_data[["T-7d"]].dropna().rename(columns={"T-7d": "forecast"})
        )

        # Maybe there is still missing data, for example if the cdb has been down for a
        # while in this case, use the load of 2 weeks before
        basecase_forecast = basecase_forecast.append(
            forecast_input_data[["T-14d"]]
            .dropna()
            .rename(columns={"T-14d": "forecast"})
        )
        basecase_forecast = basecase_forecast[
            np.invert(basecase_forecast.index.duplicated())
        ]

        return basecase_forecast.sort_index()
