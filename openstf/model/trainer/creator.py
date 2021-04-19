# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from openstf.enums import MLModelType
from openstf.model.trainer.xgboost.quantile import XGBQuantileModelTrainer
from openstf.model.trainer.xgboost.xgboost import XGBModelTrainer
from openstf.model.trainer.lightgbm.lightgbm import LGBModelTrainer


class ModelTrainerCreator:
    """Factory object for creating model trainer objects"""

    # Set object mapping
    MODEL_TRAINER_CONSTRUCTORS = {
        MLModelType.XGB: XGBModelTrainer,
        MLModelType.XGB_QUANTILE: XGBQuantileModelTrainer,
        MLModelType.LGB: LGBModelTrainer,
    }

    def __init__(self, pj):
        # check if model type is valid
        if pj["model"] not in [k.value for k in self.MODEL_TRAINER_CONSTRUCTORS]:
            raise KeyError(f'Unknown model type: {pj["model"]}')

        self.pj = pj
        # TODO see if this can be configured more generally for example in a system yaml

    def create_model_trainer(self):
        """
        Method returns model trainer objects that can be used to train models

        Returns: Implementation of the AbstractModelTrainer class tailored to
            a specific algorithm.

        """
        model_type = MLModelType(self.pj["model"])
        model_trainer = self.MODEL_TRAINER_CONSTRUCTORS[model_type]

        return model_trainer(self.pj)
