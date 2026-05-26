"""Benchmarks baselines used by the OpenSTEF Beam benchmarking utilities.

This package exposes baseline forecasters for use in backtesting.
The OpenSTEF v4 baselines require ``openstef-models`` and ``openstef-meta``,
available via the ``baselines`` extra: ``pip install openstef-beam[baselines]``.

Import directly from the submodule::

    from openstef_beam.benchmarking.baselines.openstef4 import (
        OpenSTEF4BacktestForecaster,
        create_openstef4_preset_backtest_forecaster,
    )
"""

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
