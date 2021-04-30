from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class FallbackModel:

    def predict(self, imput_df: pd.DataFrame) -> np.array:




    @staticmethod
    def predict_fallback(forecast_index, load):
        """Make a fall back forecast
            Use historic profile of most extreme day.

            Set the value of the forecast 'quality' column to 'substituted'

        Args:
            forecast_index (pandas.DatetimeIndex): Index required for the forecast
            load (pandas.DataFrame): load

        Raises:
            RuntimeError: When the most important feature does not start with
                T-, wind or radi

        Returns:
            pandas.DataFrame: Fallback forecast DataFrame with columns:
                'forecast', 'quality'
        """
        # Check if load is completely empty
        if len(load.dropna()) == 0:
            raise ValueError("No historic load data available")

        # Find most extreme historic day (do not count today as it is incomplete)
        rel_date = (
            load[load.index.tz_localize(None).date != datetime.utcnow().date()]
                .idxmax()
                .load.date()
        )
        dayprof = load[str(rel_date)].copy()
        dayprof["time"] = dayprof.index.time

        forecast = pd.DataFrame(index=forecast_index)
        forecast["time"] = forecast.index.time
        forecast = (
            forecast.reset_index()
                .merge(dayprof, left_on="time", right_on="time", how="outer")
                .set_index("index")
        )
        forecast = forecast[["load"]].rename(columns=dict(load="forecast"))

        return forecast["load"].to_numpy()