# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecasting model adapter layer for backtesting integration.

Bridges the gap between any forecasting model and the backtesting pipeline
by defining standard interfaces for training, prediction, and configuration.
"""

from openstef_beam.backtesting.backtest_forecaster.dummy_forecaster import (
    DummyForecaster,
)
from openstef_beam.backtesting.backtest_forecaster.mixins import (
    BacktestBatchForecasterMixin,
    BacktestForecasterConfig,
    BacktestForecasterMixin,
)
from openstef_beam.backtesting.backtest_forecaster.openstef4_backtest_forecaster import (
    OpenSTEF4BacktestForecaster,
)

__all__ = [
    "BacktestBatchForecasterMixin",
    "BacktestForecasterConfig",
    "BacktestForecasterMixin",
    "DummyForecaster",
    "OpenSTEF4BacktestForecaster",
]
