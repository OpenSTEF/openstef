# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Optional

import numpy as np
import pandas as pd

from openstef.model.regressors.regressor_interface import OpenstfRegressorInterface


class OpenstfRegressor(OpenstfRegressorInterface):
    def __init__(self):
        self.feature_importance_dataframe = None
        self.feature_importances_ = None

    def set_feature_importance(self) -> Optional[pd.DataFrame]:
        """get feature importance.

        Returns:
         pd.DataFrame
        """
        # returns a dict if we can get feature importance else returns None
        importance_names = self._get_importance_names()
        # if the model doesn't support feature importance return None
        if importance_names is None:
            return None

        gain = self._fraction_importance(importance_names["gain_importance_name"])
        weight_importance = self._fraction_importance(
            importance_names["weight_importance_name"]
        )

        feature_importance = pd.DataFrame(
            {"gain": gain, "weight": weight_importance}, index=self.feature_names
        )

        feature_importance.sort_values(by="gain", ascending=False, inplace=True)
        return feature_importance

    def _fraction_importance(self, importance: str) -> np.ndarray:
        self.importance_type = importance
        feature_importance = self.feature_importances_
        feature_importance = feature_importance / sum(feature_importance)
        return feature_importance

    @staticmethod
    def _get_importance_names() -> Optional[dict]:
        """Get importance names if applicable

        Returns:
            Optional (dict): Returns a dict or None, return None if the model can't get feature importance

        """
        return None
