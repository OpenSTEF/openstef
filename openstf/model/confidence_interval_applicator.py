# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
from scipy import stats


class ConfidenceIntervalApplicator:
    def __init__(self, model):
        self.confidence_interval = model.confidence_interval

    def add_confidence_interval(self, forecast, quantiles):
        temp_forecast = self._add_standard_deviation_to_forecast(forecast)
        return self._add_quantiles_to_forecast(temp_forecast, quantiles)

    def _add_standard_deviation_to_forecast(self, forecast):
        """Add a standard deviation to a live forecast.

        The stdev for intermediate forecast horizons is interpolated.

        For the input standard_deviation, it is preferred that the forecast horizon is
        expressed in Hours, instead of 'Near/Far'. For now, Near/Far is supported,
        but this will be deprecated.

        Args:
            forecast (pd.DataFrame): Forecast DataFram with columns: "forecast"
            standard_deviation (pd.DataFrame): Standard deviation. DataFrame with columns:
                "hour", "horizon", "stdev"
            (optional) interpolate (str): Interpolation method, options: "exponential" or "linear"

        Returns:
            pd.DataFrame: Forecast with added standard deviation. DataFrame with columns:
                "forecast", "stdev"
        """

        standard_deviation = self.confidence_interval

        if standard_deviation is None:
            return forecast

        # -------- Moved from feature_engineering.add_stdev ------------------------- #
        # pivot
        stdev = standard_deviation.pivot_table(columns=["horizon"], index="hour")[
            "stdev"
        ]
        # Prepare input dataframes
        # Rename Near and Far to 0.25 and 47 respectively, if present.
        # Timehorizon in hours is preferred to less descriptive Near/Far
        if "Near" in stdev.columns:
            near = (forecast.index[1] - forecast.index[0]).total_seconds() / 3600.0
            # Try to infer for forecast df, else use a max of 48 hours
            far = min(
                48.0,
                (forecast.index.max() - forecast.index.min()).total_seconds() / 3600.0,
            )
            stdev.rename(columns={"Near": near, "Far": far}, inplace=True)
        else:
            near = stdev.columns.min()
            far = stdev.columns.max()

        forecast_copy = forecast.copy()
        # add time ahead column if not already present
        if "tAhead" not in forecast_copy.columns:
            # Assume first datapoint is 'now'
            forecast_copy["tAhead"] = (
                forecast_copy.index - forecast_copy.index.min()
            ).total_seconds() / 3600.0
        # add helper column hour
        forecast_copy["hour"] = forecast_copy.index.hour

        # Define functions which can be used to approximate the error for in-between time horizons
        # Let's fit and exponential decay of accuracy
        def calc_exp_dec(t, stdev_row, near, far):
            # We use the formula sigma(t) = (1 - A * exp(-t/tau)) + b
            # Strictly speaking, tau is specific for each time series.
            # However, for simplicity, we use tau = Far/4.
            # This represents a situation where the stdev at 25% of the Far horizon,
            # has increased by two.
            tau = far / 4.0
            # Filling in the known sigma(Near) and sigma(Far) gives:
            sf, sn = stdev_row[far], stdev_row[near]
            A = (sf - sn) / ((1 - np.exp(-far / tau)) - (1 - np.exp(-near / tau)))
            b = sn - A * (1 - np.exp(-near / tau))
            return A * (1 - np.exp(-t / tau)) + b

        # Add stdev to forecast_copy dataframe
        forecast_copy["stdev"] = forecast_copy.apply(
            lambda x: calc_exp_dec(x.tAhead, stdev.loc[x.hour], near, far), axis=1
        )
        # -------- End moved from feature_engineering.add_stdev --------------------- #

        return forecast_copy.drop(columns=["hour"])

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
                forecast["forecast"] + stats.norm.ppf(quantile) * forecast["stdev"]
            )

        return forecast
