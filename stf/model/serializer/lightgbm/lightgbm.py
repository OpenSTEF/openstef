# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pickle

import lightgbm as lgb

from stf.model.serializer.serializer import AbstractModelSerializer


class LGBModelSerializer(AbstractModelSerializer):
    def __init__(self):
        super().__init__()

    def save(self, pid, lgb_model, corrections=None):

        save_folder = self.build_save_folder_path(pid=pid)

        save_folder.mkdir(parents=True, exist_ok=True)

        # save LGB model
        print(save_folder)
        lgb_model.save_model(str(save_folder / self.MODEL_FILENAME))
        # save feature names
        # with open(save_folder / self.FEATURE_NAMES_FILENAME, "w") as fh:
        #     fh.write(str(lgb_model.feature_names))
        # # save feature types
        # with open(save_folder / self.FEATURE_TYPES_FILENAME, "w") as fh:
        #     fh.write(str(lgb_model.feature_types))
        # # save best iteration / best n_trees
        # if getattr(xgb_model, "best_ntree_limit") is not None:
        #     with open(save_folder / self.BEST_ITERATION_FILENAME, "wb") as fh:
        #         pickle.dump(xgb_model.best_ntree_limit, fh)
        # else:
        #     self.logger.warning(
        #         f"No best iteration found, no best iteration save for pid: {pid}"
        #     )
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

    def _add_atributes(self, model_folder, model):

        # load feature names
        with open(model_folder / self.FEATURE_NAMES_FILENAME, "r") as fh:
            # Reformat feature names
            # TODO this is unnecessary when just saved properly (i.e yaml, json, xml, ini, pickle etc)
            model.feature_names = (
                fh.read()
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace(" ", "")
                .split(",")
            )
        # Load feature types
        with open(model_folder / self.FEATURE_TYPES_FILENAME, "r") as fh:
            # Reformat feature types
            model.feature_types = (
                fh.read()
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace(" ", "")
                .split(",")
            )

        # Load best iteration (if exist)
        best_iteration_filepath = model_folder / self.BEST_ITERATION_FILENAME
        if best_iteration_filepath.is_file() is True:
            with open(best_iteration_filepath, "rb") as fh:
                model.best_ntree_limit = pickle.load(fh)
        else:
            self.logger.warning(
                f"Could not load best iteration, file does not exists '{best_iteration_filepath}'"
            )
            # TODO magic number?
            model.best_ntree_limit = 15
        return model
