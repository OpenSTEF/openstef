from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from openstf.enums import MLModelType


class ModelCreator:
    """Factory object for creating model trainer objects"""

    # Set object mapping
    MODEL_TRAINER_CONSTRUCTORS = {
        MLModelType.XGB: XGBRegressor,
        MLModelType.LGB: LGBMRegressor,
    }

    def __init__(self, pj):
        # check if model type is valid
        if pj["model"] not in [k.value for k in self.MODEL_TRAINER_CONSTRUCTORS]:
            raise KeyError(f'Unknown model type: {pj["model"]}')

        self.pj = pj
        # TODO see if this can be configured more generally for example in a system yaml

    def create_model(self):

        model_type = MLModelType(self.pj["model"])

        return self.MODEL_TRAINER_CONSTRUCTORS[model_type]()
