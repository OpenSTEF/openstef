# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import structlog

from openstf.model.prediction.xgboost.model.quantile import XGBQuantileModel
from openstf.model.serializer.xgboost.xgboost import XGBModelSerializer


class XGBQuantileModelSerializer(XGBModelSerializer):
    def __init__(self):
        super().__init__()
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.MODEL_FILENAME = "model_quantile.bin"

    def load(self, pid, pid_model_folder=None):
        """Load the most recent model

        Args:
            pid ([type]): Prediction job id.
            pid_model_folder(str): Path to save model at specific location

        Raises:
            FileNotFoundError: If no recent model file was found

        Returns:
            tuple:
                [0]: Model
                [1]: Path to model file
        """
        if pid_model_folder is None:
            pid_model_folder = self.build_pid_model_folder_path(pid=pid)

        try:
            model_file, model_folder = self.find_most_recent_model_file(
                pid_model_folder
            )
        except FileNotFoundError as e:
            self.logger.error(f"Can't load model, no recent model file found. {e}")
            return None, None

        self.logger.info(f"Loaded model from: {model_file}", model_file=model_file)

        xgb_quantile_model = XGBQuantileModel(model_file=model_file)

        xgb_quantile_model = self._add_atributes(model_folder, xgb_quantile_model)

        return xgb_quantile_model, model_file
