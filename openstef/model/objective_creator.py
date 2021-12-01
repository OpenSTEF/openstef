# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union

from openstef.enums import MLModelType
from openstef.model.objective import (
    LinearRegressorObjective,
    LGBRegressorObjective,
    ProLoafRegressorObjective,
    RegressorObjective,
    XGBQuantileRegressorObjective,
    XGBRegressorObjective,
)
from openstef.model.regressors.custom_regressor import (
    create_custom_objective,
    is_custom_type,
)


class ObjectiveCreator:
    OBJECTIVES = {
        MLModelType.XGB: XGBRegressorObjective,
        MLModelType.LGB: LGBRegressorObjective,
        MLModelType.XGB_QUANTILE: XGBQuantileRegressorObjective,
        MLModelType.ProLoaf: ProLoafRegressorObjective,
        MLModelType.LINEAR: LinearRegressorObjective,
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
            if is_custom_type(model_type):
                objective = create_custom_objective
            else:
                model_type = MLModelType(model_type)
                objective = ObjectiveCreator.OBJECTIVES[model_type]
        except ValueError as e:
            valid_types = [t.value for t in MLModelType]
            raise NotImplementedError(
                f"No objective for '{model_type}', "
                f"valid model_types are: {valid_types}"
                f"or import a custom model"
            ) from e

        return objective
