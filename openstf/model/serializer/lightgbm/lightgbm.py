# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import lightgbm as lgb

from openstf.model.serializer.serializer import AbstractModelSerializer


class LGBModelSerializer(AbstractModelSerializer):
    def __init__(self):
        super().__init__()

    def save(self, pid, lgb_model, corrections=None):

        save_folder = self.build_save_folder_path(pid=pid)

        save_folder.mkdir(parents=True, exist_ok=True)

        # save LGB model
        self.logger.warning(f"Save folder: {save_folder}")
        lgb_model.save_model(str(save_folder / self.MODEL_FILENAME))

        # save corrections
        if corrections is not None:
            if len(corrections) == len(corrections.dropna()):
                corrections.to_csv(
                    save_folder / self.CORRECTIONS_FILENAME, **self.PANDAS_CSV_FORMAT
                )
            return

        self.logger.warning(
            "No corrections found, corrections file not saved for pid: {pid}"
        )

    # NOTE from general.py
    def load(self, pid, pid_model_folder=None):
        """Load the most recent model.

        Args:
            pid (int): Prediction job id
            pid_model_folder(str): Path to save model at specific location

        Raises:
            FileNotFoundError: If no recent model file was found

        Returns:
            tuple: Tuple with:
                [0]: lgb.Booster: Loaded model
                [1]: str: Path to loaded model
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
        model_file = str(model_file)
        lgb_model = lgb.Booster(model_file=model_file)
        self.logger.info(f"Loaded model from: {model_file}")

        return lgb_model, model_file
