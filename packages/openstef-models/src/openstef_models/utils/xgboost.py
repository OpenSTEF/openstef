# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""XGBoost-specific utility helpers."""

import numpy as np
import pandas as pd

from openstef_core.types import Quantile

try:
    import xgboost as xgb
except ImportError as e:
    from openstef_core.exceptions import MissingExtraError

    raise MissingExtraError("xgboost", "openstef-models") from e


def get_median_shap_contribs(
    booster: xgb.Booster,
    input_data: pd.DataFrame,
    quantiles: list[Quantile],
) -> np.ndarray:
    """Compute SHAP contributions and return the median-quantile slice.

    Args:
        booster: Fitted XGBoost Booster.
        input_data: Feature matrix for which to compute contributions.
        quantiles: Quantiles the model was trained on.

    Returns:
        Array of shape ``(n_samples, n_features + 1)`` where the last column is
        the model bias.
    """
    dmatrix = xgb.DMatrix(input_data)
    contribs_raw: np.ndarray = booster.predict(dmatrix, pred_contribs=True)
    n_samples = len(input_data)
    n_quantiles = len(quantiles)
    contribs_3d = contribs_raw.reshape(n_samples, n_quantiles, -1)
    median_idx = min(range(n_quantiles), key=lambda i: abs(float(quantiles[i]) - 0.5))
    return contribs_3d[:, median_idx, :].copy()


__all__ = ["get_median_shap_contribs"]
