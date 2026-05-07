# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Optuna-powered hyperparameter tuning integration.

Requires ``optuna`` (install via ``pip install openstef-models[tuning]``).

Public API
----------
- `HyperparameterTuner` — orchestrates Bayesian tuning over HyperParams search spaces.
- `TuningResult` — result container with best config and Optuna study.
"""

from openstef_models.integrations.optuna.tuner import (
    HyperparameterTuner,
    TuningResult,
)

__all__ = [
    "HyperparameterTuner",
    "TuningResult",
]
