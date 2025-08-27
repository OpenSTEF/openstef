# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests forecasting models by simulating how they would perform in real operations.

In real-world energy forecasting, you can't use future data to make predictions about the past.
This module creates realistic test scenarios by "replaying" historical data as if it were
happening in real-time. Models are retrained periodically, just like in actual deployment,
and predictions are made with only past data available.

Why this matters:
    - Prevents data leakage: No cheating with future information
    - Realistic performance: Results match what you'd see in production
    - Operational constraints: Simulates real training schedules and data availability
    - Fair comparison: All models tested under identical conditions
"""

from openstef_beam.backtesting import backtest_forecaster
from openstef_beam.backtesting.backtest_pipeline import BacktestConfig, BacktestPipeline

__all__ = ["BacktestConfig", "BacktestPipeline", "backtest_forecaster"]
