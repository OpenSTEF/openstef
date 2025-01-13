# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union

from openstef.enums import ModelType
from openstef.model.objective import (
    ARIMARegressorObjective,
    LGBRegressorObjective,
    LinearRegressorObjective,
    RegressorObjective,
    XGBQuantileRegressorObjective,
    XGBRegressorObjective,
    XGBMultioutputQuantileRegressorObjective,
)
from openstef.model.regressors.custom_regressor import (
    create_custom_objective,
    is_custom_type,
)


class ObjectiveCreator:
    OBJECTIVES = {
        ModelType.XGB: XGBRegressorObjective,
        ModelType.LGB: LGBRegressorObjective,
        ModelType.XGB_QUANTILE: XGBQuantileRegressorObjective,
        ModelType.XGB_MULTIOUTPUT_QUANTILE: XGBMultioutputQuantileRegressorObjective,
        ModelType.LINEAR: LinearRegressorObjective,
        ModelType.LINEAR_QUANTILE: LinearRegressorObjective,
        ModelType.ARIMA: ARIMARegressorObjective,
    }

    @staticmethod
    def create_objective(model_type: Union[ModelType, str]) -> RegressorObjective:
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
                model_type = ModelType(model_type)
                objective = ObjectiveCreator.OBJECTIVES[model_type]
        except ValueError as e:
            valid_types = [t.value for t in ModelType]
            raise NotImplementedError(
                f"No objective for '{model_type}', "
                f"valid model_types are: {valid_types}"
                "or import a custom model"
            ) from e

        return objective
