# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Custom Forecaster Template
#
# Implements [`BacktestForecasterMixin`](https://openstef.github.io/openstef/api/generated/openstef_beam.backtesting.BacktestForecasterMixin.html)
# — the interface BEAM needs to run any model in its backtesting/benchmarking pipeline.
#
# **User story:** *"I want to benchmark my own model."*
#
# Copy this file and modify `fit()` and `predict()` to wrap your model.
#
# **See also:**
# - [BacktestForecasterConfig](https://openstef.github.io/openstef/api/generated/openstef_beam.backtesting.BacktestForecasterConfig.html) — scheduling settings
# - [RestrictedHorizonVersionedTimeSeries](https://openstef.github.io/openstef/api/generated/openstef_beam.backtesting.RestrictedHorizonVersionedTimeSeries.html) — the data view passed to `fit()` and `predict()`
# - [Backtesting quickstart tutorial](../../tutorials/backtesting_quickstart.ipynb) — introduction to backtesting concepts

# %% tags=["remove-cell"]
"""Custom baseline: predicts a constant value (last known median) for all future timestamps.

Implements BacktestForecasterMixin — the interface BEAM needs to run any model
in its backtesting/benchmarking pipeline. To create your own baseline, copy this
file and modify fit() and predict().
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# %%

from datetime import timedelta
from typing import override

import pandas as pd

from openstef_beam.backtesting.backtest_forecaster import BacktestForecasterConfig, BacktestForecasterMixin
from openstef_beam.backtesting.restricted_horizon_timeseries import RestrictedHorizonVersionedTimeSeries
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import Q, Quantile

# %% [markdown]
# ## The `BacktestForecasterMixin` interface
#
# Your forecaster must implement:
# - `config` — a [`BacktestForecasterConfig`](https://openstef.github.io/openstef/api/generated/openstef_beam.backtesting.BacktestForecasterConfig.html) that tells BEAM how to schedule training and prediction
# - `quantiles` — which probabilistic bands to produce (e.g. `[Q(0.05), Q(0.5), Q(0.95)]`)
# - `fit(data)` — train your model on restricted-horizon data (no lookahead)
# - `predict(data)` → `TimeSeriesDataset | None` — produce a forecast
#
# BEAM calls `fit()` at `train_interval` spacing, and `predict()` at `predict_interval` spacing.

# %%


class ExampleBenchmarkForecaster(BacktestForecasterMixin):
    """Predicts a constant median of recent history for all future timestamps.

    All quantile columns get the same value -- no uncertainty estimation.
    """

    def __init__(self, predict_quantiles: list[Quantile] | None = None) -> None:  # noqa: D107
        # Quantiles define the probabilistic forecast bands (e.g. P05 = 5th percentile)
        self._quantiles = predict_quantiles or [Q(0.05), Q(0.1), Q(0.3), Q(0.5), Q(0.7), Q(0.9), Q(0.95)]
        self._median: float = 0.0

        # BacktestForecasterConfig tells BEAM how to schedule training and prediction
        self.config = BacktestForecasterConfig(
            requires_training=True,  # Call fit() before predict()
            predict_length=timedelta(days=7),  # How far ahead to forecast
            predict_min_length=timedelta(minutes=15),
            predict_context_length=timedelta(minutes=15),  # Data needed before forecast start (>0)
            predict_context_min_coverage=0.0,
            training_context_length=timedelta(days=30),  # How much history fit() sees
            training_context_min_coverage=0.3,  # Min 30% non-NaN data required
            predict_sample_interval=timedelta(minutes=15),  # Output resolution (15-min intervals)
        )

    @property
    @override
    def quantiles(self) -> list[Quantile]:
        """Quantiles this forecaster produces."""
        return self._quantiles

    @override
    def fit(self, data: RestrictedHorizonVersionedTimeSeries) -> None:
        """Compute median of recent load data.

        Args:
            data: Restricted-horizon view -- only sees data available at data.horizon.
        """
        # data.horizon = the current point in time during backtesting
        # get_window() returns only data that was available at that point (no lookahead)
        training = data.get_window(
            start=data.horizon - self.config.training_context_length,  # 30 days before horizon
            end=data.horizon,
            available_before=data.horizon,  # Ensures no future data leaks in
        )
        # "load" is the target column (actual energy consumption/generation)
        if "load" in training.data.columns:
            self._median = float(training.data["load"].median())

    @override
    def predict(self, data: RestrictedHorizonVersionedTimeSeries) -> TimeSeriesDataset | None:
        """Return constant median prediction for the forecast horizon.

        Returns:
            Forecast with all quantiles set to the training median, or None on failure.
        """
        # Build a DataFrame with "load" + one column per quantile (e.g. "quantile_P05")
        # All values are the same constant (the median from fit())
        # q.format() converts Q(0.05) -> "quantile_P05" (the required column naming)
        return TimeSeriesDataset(
            data=pd.DataFrame(
                data={"load": self._median} | {q.format(): self._median for q in self._quantiles},
                index=pd.DatetimeIndex(
                    pd.date_range(
                        data.horizon,
                        periods=int(self.config.predict_length / self.config.predict_sample_interval),
                        freq=self.config.predict_sample_interval,
                    ),
                    name="datetime",
                ),
            ),
            sample_interval=self.config.predict_sample_interval,
        )
