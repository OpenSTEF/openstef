# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from datetime import datetime

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sklearn.base import RegressorMixin

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.exceptions import ModelWithoutStDev


class ConfidenceIntervalApplicator:
    def __init__(self, model: RegressorMixin, forecast_input_data: pd.DataFrame):
        self.model = model
        self.forecast_input_data = forecast_input_data
        self.logger = structlog.get_logger(self.__class__.__name__)

    def add_confidence_interval(
        self,
        forecast: pd.DataFrame,
        pj: PredictionJobDataClass,
    ) -> pd.DataFrame:
        """Add a confidence interval to a forecast.

        Adds a confidence interval to a forecast in two ways:
            1. "stdev" column, this is a column with a standard deviation that is
                determined during training (ConfidenceGenerator)
            2. Quantile columns, these columns give a more precise defenition of the
                confidence interval. Quantile columns are determined with one of two
                methods, depending on the model type group:

                a. Default, using the "stdev" column and the assumption the error is
                    normally distributed.
                b. Quantile regression, this method is only available for quantile
                    models and uses specifically trained models to estimate the
                    quantiles of the confidence interval.

                Depending on the model type (quantile or non quantile),
                 a confidence interval is added to the forecast based on quantile
                 regression or the default method.

        Args:
            forecast: Forecast DataFrame with columns: "forecast"
            pj: Prediction job

        Returns:
            Forecast DataFrame with columns; "forecast", "stdev" and quantile columns.

        """
        temp_forecast = self._add_standard_deviation_to_forecast(forecast)

        if self.model.can_predict_quantiles:
            return self._add_quantiles_to_forecast_quantile_regression(
                temp_forecast, pj["quantiles"]
            )

        return self._add_quantiles_to_forecast_default(temp_forecast, pj["quantiles"])

    def _add_standard_deviation_to_forecast(
        self, forecast: pd.DataFrame
    ) -> pd.DataFrame:
        """Add a standard deviation to a live forecast.

        The stdev for intermediate forecast horizons is interpolated.

        Args:
            forecast: Forecast DataFrame with columns: "forecast"

        Returns:
            Forecast with added standard deviation. DataFrame with columns:
                "forecast", "stdev"

        """
        minimal_resolution: int = 15  # Minimal time resolution in minutes
        standard_deviation = self.model.standard_deviation

        # raise an exception if no valid standard deviation is available
        if standard_deviation is None:
            raise ModelWithoutStDev("No stdev available")

        if standard_deviation.empty:  # make separate statement to avoid None.empty
            raise ModelWithoutStDev("No stdev available")

        if standard_deviation.stdev.isnull().values.all():
            raise ModelWithoutStDev("All stdev values are NA")

        # Fill stdev nans with the mean of all stdev values
        if standard_deviation.stdev.isnull().values.any():
            self.logger.warning(
                "Stdev for some hours is not known, filling in with mean."
            )
            standard_deviation["stdev"] = standard_deviation.stdev.fillna(
                standard_deviation.stdev.mean()
            )

        # pivot to have a dataframe with columns [stdev, hour, horizon] for a
        # 'near' and a 'far' horizon
        stdev = standard_deviation.pivot_table(columns=["horizon"], index="hour")[
            "stdev"
        ]
        # Prepare input dataframes for near and far horizon
        near = stdev.columns.min()
        far = stdev.columns.max()

        forecast_copy = forecast.copy()
        # add time ahead column if not already present
        if "tAhead" not in forecast_copy.columns:
            # Determine now, rounded on 15 minutes,
            # Rounding helps to prevent fractional t_aheads
            now = (
                pd.Series(datetime.utcnow().replace(tzinfo=forecast_copy.index.tzinfo))
                .min()
                .round(f"{minimal_resolution}T")
                .to_pydatetime()
            )
            # Determine t_aheads by subtracting with now
            forecast_copy["tAhead"] = (
                forecast_copy.index - now
            ).total_seconds() / 3600.0

        # add helper column hour
        forecast_copy["hour"] = forecast_copy.index.hour

        # Define functions which can be used to approximate the error for in-between
        # time horizons
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
            value = A * (1 - np.exp(-t / tau)) + b
            # cap the value to keep it between near and far
            if value < sn:
                return sn
            return sf if value > sf else value

        # If only one horizon is available use that one
        if len(stdev.columns) == 1:
            forecast_copy["stdev"] = forecast_copy.apply(
                lambda x: stdev.loc[x.hour], axis=1
            )
        # If more are available do something fancy with interpolation
        else:
            # Add stdev to forecast_copy dataframe
            forecast_copy["stdev"] = forecast_copy.apply(
                lambda x: calc_exp_dec(x.tAhead, stdev.loc[x.hour], near, far), axis=1
            )
        return forecast_copy.drop(columns=["hour"])

    @staticmethod
    def _add_quantiles_to_forecast_default(
        forecast: pd.DataFrame, quantiles: list[float]
    ) -> pd.DataFrame:
        """Add quantiles to forecast.

            Use the standard deviation to calculate the quantiles.

        Args:
            forecast: Forecast (should contain a 'forecast' + 'stdev' column)
            quantiles: List with desired quantiles,
                for example: [0.01, 0.1, 0.9, 0.99]

        Returns:
            Forecast DataFrame with quantile (e.g. 'quantile_PXX')
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

    def _add_quantiles_to_forecast_quantile_regression(
        self, forecast: pd.DataFrame, quantiles: list[float]
    ) -> pd.DataFrame:
        """Add quantiles to forecast.

            Use trained quantile regression model to calculate the quantiles.

        Args:
            forecast: Forecast
            quantiles: List with desired quantiles

        Returns:
            Forecast DataFrame with quantile (e.g. 'quantile_PXX')
                columns added.

        """
        # Only determine quantiles for datetimes in forecast
        quantile_df = pd.DataFrame(index=self.forecast_input_data.index)
        for quantile in quantiles:
            quantile_key = f"quantile_P{quantile * 100:02.0f}"
            quantile_df[quantile_key] = self.model.predict(
                self.forecast_input_data, quantile=quantile
            )

        return forecast.merge(
            quantile_df, left_index=True, right_index=True, how="left"
        )
