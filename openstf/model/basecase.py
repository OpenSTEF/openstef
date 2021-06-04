from datetime import datetime, timedelta, timezone
import pytz
import numpy as np
import pandas as pd


from sklearn.base import BaseEstimator, RegressorMixin

MINIMAL_RESOLUTION: int = 15  # Used for validating the forecast input


class BaseCaseModel(BaseEstimator, RegressorMixin):
    def predict(self, forecsst_input_data: pd.DataFrame) -> pd.DataFrame:
        """Predict using the basecase method

        Args:
            forecsst_input_data (pd.DataFrame): Forecast input dataframe

        Returns: pd.DataFrame: Basecase forecast

        """
        return self.make_basecase_forecast(forecsst_input_data)

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
            ValueError: In case T-7d or T-14d is not present a ValueError is raised.
            Also in the case the start of the forecast is before the horizon of the regular forecast,
             a value error is raised
        Returns:
            pd.DataFrame: Basecase forecast


        """

        # Validate input to make sure we are not overwriting regular forecasts
        requested_start = forecast_input_data.index.min().ceil(f"{MINIMAL_RESOLUTION}T")
        allowed_start = pd.Series(
            datetime.utcnow().replace(tzinfo=pytz.utc)
        ).min().floor(f"{MINIMAL_RESOLUTION}T").to_pydatetime() + timedelta(
            hours=overwrite_delay_hours
        )
        if requested_start < allowed_start:
            raise ValueError(
                f"Basecase forecast requested for horizon of regular forecast! Please check input! Requested start {requested_start}, allowed start {allowed_start}"
            )

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

        # Also make a basecase forecast for the forecast_other component. This will make a
        # simple basecase components forecast available and ensures that the sum of
        # the components (other, wind and solar) is equal to the normal basecase forecast
        # This is important for sending GLMD messages correctly to TenneT!
        basecase_forecast["forecast_other"] = basecase_forecast["forecast"]

        return basecase_forecast.sort_index()
