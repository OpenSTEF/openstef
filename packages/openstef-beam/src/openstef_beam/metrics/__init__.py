# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Metrics for measuring how well energy forecasting models perform.

Energy forecasting models can predict either single values (deterministic) or probability
distributions (probabilistic). This module provides specialized metrics for both types,
with particular focus on the operational challenges of energy systems like peak detection
and scale differences between high and low load periods.

What you can measure:
    - Forecast accuracy: How close are predictions to actual values?
    - Peak detection: Can the model identify congestion events?
    - Uncertainty quality: For probabilistic forecasts, are confidence intervals reliable?
    - Cross-period comparison: Compare model performance across seasons, times, etc.
"""

from openstef_beam.metrics.metrics_deterministic import (
    ConfusionMatrix,
    PrecisionRecall,
    confusion_matrix,
    fbeta,
    mape,
    precision_recall,
    r2,
    relative_pinball_loss,
    riqd,
    rmae,
)
from openstef_beam.metrics.metrics_probabilistic import (
    crps,
    mean_absolute_calibration_error,
    observed_probability,
    rcrps,
)

__all__ = [
    "ConfusionMatrix",
    "PrecisionRecall",
    "confusion_matrix",
    "crps",
    "fbeta",
    "mape",
    "mean_absolute_calibration_error",
    "observed_probability",
    "precision_recall",
    "r2",
    "rcrps",
    "relative_pinball_loss",
    "riqd",
    "rmae",
]
