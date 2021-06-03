from datetime import datetime, timedelta, timezone

import numpy as np


from sklearn.base import BaseEstimator, RegressorMixin


class BaseCaseModel(BaseEstimator, RegressorMixin):
    def predict(self, forecsst_input_data):
        return self.make_basecase_forecast(forecsst_input_data)

    @staticmethod
    def make_basecase_forecast(forecast_input_data, overwrite_delay_hours=48):
        """Make a 'basecase' forecast
            Result is writen to database for all forecasts further in time than
            overwrite_delay_hours. Idea is that if all else fails, this forecasts is
            still available.
            Basecase example: the load of last week.
            As 'quality', the value 'not_renewed' is used.
        Args:
            forecast_input_data (pd.DataFrame): Forecast input dataframe
            overwrite_delay_hours (float): times before this in the future are not
                forecasted
        Returns:
            pd.DataFrame: Basecase forecast (which was written to the database)
        """

        # - Make forecast
        # Make basecase forecast: Use load of last week
        basecase_forecast = forecast_input_data[['T-7d']].dropna().rename(columns={'T-7d':'forecast'})



        # Maybe there is still missing data, for example if the cdb has been down for a
        # while in this case, use the load of 2 weeks before
        basecase_forecast = basecase_forecast.append(forecast_input_data[['T-14d']].dropna().rename(columns={'T-14d':'forecast'}))
        basecase_forecast = basecase_forecast[
            np.invert(basecase_forecast.index.duplicated())
        ]

        # Don't update first 48 hours
        forecast_start = datetime.now(timezone.utc) + timedelta(
            hours=overwrite_delay_hours
        )
        basecase_forecast = basecase_forecast[forecast_start:]

        # Also make a basecase forecast for the forecast_other component. This will make a
        # simple basecase components forecast available and ensures that the sum of
        # the components (other, wind and solar) is equal to the normal basecase forecast
        basecase_forecast["forecast_other"] = basecase_forecast["forecast"]

        return basecase_forecast.sort_index()





