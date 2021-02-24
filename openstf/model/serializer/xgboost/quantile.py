# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from ktpbase.log import logging

from openstf.model.prediction.xgboost.model.quantile import XGBQuantileModel
from openstf.model.serializer.xgboost.xgboost import XGBModelSerializer


class XGBQuantileModelSerializer(XGBModelSerializer):
    def __init__(self):
        super().__init__()
        self.logger = logging.get_logger(self.__class__.__name__)
        self.MODEL_FILENAME = "model_quantile.bin"

    # NOTE from general.py
    def load(self, pid):
        """Load the most recent model

        Args:
            pid ([type]): [description]
            forecast_type ([type]): [description]

        Raises:
            FileNotFoundError: If no recent model file was found

        Returns:
            [type]: [description]
        """

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
