# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Optuna-powered hyperparameter tuning integration.

Requires ``optuna`` (install via ``pip install openstef-models[tuning]``).

Public API
----------
- `HyperparameterTuner` — orchestrates Bayesian tuning over HyperParams search spaces.
- `TuningResult` — result container with best config and Optuna study.
- `apply_trial_suggestions` — standalone utility for applying Optuna suggestions to a config.
"""

from openstef_models.integrations.optuna.tuner import (
    HyperparameterTuner,
    TuningResult,
    apply_trial_suggestions,
)

__all__ = [
    "HyperparameterTuner",
    "TuningResult",
    "apply_trial_suggestions",
]
