# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union

from openstf.enums import MLModelType
from openstf.model.objective import RegressorObjective, XGBRegressorObjective


class ObjectiveCreator:
    OBJECTIVES = {MLModelType.XGB.value: XGBRegressorObjective}

    @staticmethod
    def create_objective(model_type: Union[MLModelType, str]) -> RegressorObjective:
        valid_types = list(ObjectiveCreator.OBJECTIVES.keys())
        if model_type not in valid_types:
            raise NotImplementedError(
                f"No objective function for {model_type} valid model_types are:"
                f"{', '.join([t for t in valid_types])}"
            )
        model_type = MLModelType(model_type).value

        return ObjectiveCreator.OBJECTIVES[model_type]
