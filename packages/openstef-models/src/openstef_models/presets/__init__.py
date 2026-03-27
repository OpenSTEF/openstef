# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
# pyright: reportUnsupportedDunderAll=false

"""Forecasting workflow presets package.

Provides configurations and utilities for setting up forecasting workflows.
Tuning symbols (``HyperparameterTuner``, ``TuningResult``, ``tune``,
``fit_with_tuning``) are re-exported lazily so that ``optuna`` is only
imported when actually accessed.
"""

from typing import Any

from .forecasting_workflow import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)

_TUNING_SYMBOLS = frozenset({"HyperparameterTuner", "TuningResult", "fit_with_tuning", "tune"})

__all__ = [  # noqa: F822
    "ForecastingWorkflowConfig",
    "HyperparameterTuner",
    "TuningResult",
    "create_forecasting_workflow",
    "fit_with_tuning",
    "tune",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name in _TUNING_SYMBOLS:
        from openstef_models.utils import tuning  # noqa: PLC0415

        return getattr(tuning, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
