# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union

from openstf.enums import MLModelType
from openstf.model.objective import (
    RegressorObjective,
    XGBRegressorObjective,
    LGBRegressorObjective,
    XGBQuantileRegressorObjective,
)


class ObjectiveCreator:
    OBJECTIVES = {
        MLModelType.XGB: XGBRegressorObjective,
        MLModelType.LGB: LGBRegressorObjective,
        MLModelType.XGB_QUANTILE: XGBQuantileRegressorObjective,
    }

    @staticmethod
    def create_objective(model_type: Union[MLModelType, str]) -> RegressorObjective:
        """Create an objective function based on model type.
        Args:
            model_type (Union[MLModelType, str]): Model type to construct.
        Raises:
            NotImplementedError: When using an invalid model_type.
        Returns:
            RegressorObjective: Objective function
        """
        try:
            # This will raise a ValueError when an invalid model_type str is used
            # and nothing when a MLModelType enum is used.
            model_type = MLModelType(model_type)
        except ValueError as e:
            valid_types = [t.value for t in MLModelType]
            raise NotImplementedError(
                f"No objective for '{model_type}', "
                f"valid model_types are: {valid_types}"
            ) from e

        return ObjectiveCreator.OBJECTIVES[model_type]
