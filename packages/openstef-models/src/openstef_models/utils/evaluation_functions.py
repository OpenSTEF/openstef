# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Utility functions for evaluation metrics in forecasting models."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np

from openstef_beam.metrics.metrics_probabilistic import mean_pinball_loss
from openstef_core.types import Quantile

type EvaluationFunctionType = Literal["mean_pinball_loss"]

EVALUATION_MAP = {
    "mean_pinball_loss": mean_pinball_loss,
}


def get_evaluation_function(
    function_type: EvaluationFunctionType, quantiles: list[Quantile] | None = None, **kwargs: Any
) -> Callable[[np.ndarray, np.ndarray], float]:
    eval_metric = partial(EVALUATION_MAP[function_type], quantiles=quantiles, **kwargs)
    eval_metric.__name__ = function_type  # pyright: ignore[reportAttributeAccessIssue]
    return eval_metric


__all__ = ["EVALUATION_MAP", "EvaluationFunctionType"]
