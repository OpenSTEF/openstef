

from typing import Union

from openstf.enums import MLModelType
from openstf.model.objective import XGBRegressorObjective


class ObjectiveCreator:
    OBJECTIVES = {
        MLModelType.XGB: XGBRegressorObjective
    }
    @staticmethod
    def create_objective(model_type: Union[MLModelType, str]):
        model_type = MLModelType(model_type)

        return ObjectiveCreator.OBJECTIVES[model_type]
