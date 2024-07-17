# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Optional

import numpy as np
from sklearn.base import RegressorMixin

from xgboost import XGBRegressor

from openstef.model.regressors.regressor import OpenstfRegressor


class XGBOpenstfRegressor(XGBRegressor, OpenstfRegressor):
    """XGB Regressor which implements the Openstf regressor API."""

    gain_importance_name = "total_gain"
    weight_importance_name = "weight"

    @property
    def feature_names(self):
        return self._Booster.feature_names

    @property
    def can_predict_quantiles(self):
        return False

    @staticmethod
    def _get_importance_names():
        return {
            "gain_importance_name": "total_gain",
            "weight_importance_name": "weight",
        }

    def fit(
        self,
        x: np.array,
        y: np.array,
        *,
        early_stopping_rounds: Optional[int] = None,
        callbacks: Optional[list] = None,
        eval_metric: Optional[str] = None,
        **kwargs
    ):
        if early_stopping_rounds is not None:
            self.set_params(early_stopping_rounds=early_stopping_rounds)
        if callbacks is not None:
            self.set_params(callbacks=callbacks)
        if eval_metric is not None:
            self.set_params(eval_metric=eval_metric)

        super().fit(x, y, **kwargs)
