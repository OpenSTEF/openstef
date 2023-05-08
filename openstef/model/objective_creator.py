# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union

from openstef.enums import MLModelType
from openstef.model.objective import (
    LGBRegressorObjective,
    LinearRegressorObjective,
    ProLoafRegressorObjective,
    RegressorObjective,
    XGBQuantileRegressorObjective,
    XGBRegressorObjective,
    ARIMARegressorObjective,
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
        MLModelType.ARIMA: ARIMARegressorObjective,
    }

    @staticmethod
    def create_objective(model_type: Union[MLModelType, str]) -> RegressorObjective:
        """Create an objective function based on model type.

        Args:
            model_type: Model type to construct.

        Raises:
            NotImplementedError: When using an invalid model_type.

        Returns:
            Objective function

        """
        try:
            # This will raise a ValueError when an invalid model_type str is used
            # and nothing when a MLModelType enum is used.
            if is_custom_type(model_type):
                objective = create_custom_objective(model_type)
            else:
                model_type = MLModelType(model_type)
                objective = ObjectiveCreator.OBJECTIVES[model_type]
        except ValueError as e:
            valid_types = [t.value for t in MLModelType]
            raise NotImplementedError(
                f"No objective for '{model_type}', "
                f"valid model_types are: {valid_types}"
                "or import a custom model"
            ) from e

        return objective
