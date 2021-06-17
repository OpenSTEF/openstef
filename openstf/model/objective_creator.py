# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Union

from openstf.enums import MLModelType
from openstf.model.objective import RegressorObjective, XGBRegressorObjective


class ObjectiveCreator:
    OBJECTIVES = {MLModelType.XGB: XGBRegressorObjective}

    @staticmethod
    def create_objective(model_type: Union[MLModelType, str]) -> RegressorObjective:
        model_type = MLModelType(model_type)

        return ObjectiveCreator.OBJECTIVES[model_type]
