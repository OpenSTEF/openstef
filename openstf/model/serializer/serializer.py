# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import structlog
from ktpbase.config.config import ConfigManager


class AbstractModelSerializer(ABC):

    MODEL_FILENAME = "model.bin"
    FOLDER_DATETIME_FORMAT = "%Y%m%d%H%M%S"
    CORRECTIONS_FILENAME = "corrections.csv"
    FEATURE_NAMES_FILENAME = "featurenames.txt"
    FEATURE_TYPES_FILENAME = "featuretypes.txt"
    BEST_ITERATION_FILENAME = "best_iteration.pkl"
    PANDAS_CSV_FORMAT = {"decimal": ".", "sep": ","}

    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.trained_models_folder = Path(
            ConfigManager.get_instance().paths.trained_models
        )

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        # remove logger since its not pickable
        state.pop("logger")
        # what we return here will be stored in the pickle
        return state

    def __setstate__(self, newstate):
        # obtain a new logger
        newstate["logger"] = structlog.get_logger(self.__class__.__name__)
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(newstate)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    def build_pid_model_folder_path(self, pid, custom_folder=None):
        """Build the path to the folder where trained models for the given pid are stored.

            The trainded models are stored a folder structure using the following
            template: <trained_models_folder>/<pid>[_<component-name>]/<YYYYMMDDHHMMSS>/

        Args:
            pid (int): Prediction id
            custom_folder (pathlike): Path to custom trainded models folder
        """

        model_folder_name = f"{pid}"

        if custom_folder is not None:
            trained_models_folder = custom_folder
        else:
            trained_models_folder = self.trained_models_folder

        model_folder = trained_models_folder / model_folder_name

        return model_folder

    def find_most_recent_model_file(self, pid_model_folder):
        """Find the model recent stored model file.

            Iterate over the directories in the 'pid' model folder (the top level model
            folder for a specific pid) and find the

        Args:
            pid_model_folder (pathlib.Path): [description]

        Raises:
            FileNotFoundError: When no model file is found in any of the subdirectories
                of `pid_model_folder` which start with a 2.

        Returns:
            pathlib.Path: Path to the most recent model file
        """

        model_folders = []

        # Loop over all subfolder of pid_model_folder and store the ones starting with
        # a 2 (date starts with 2000s)
        for folder in pid_model_folder.iterdir():
            if folder.name.startswith("2") is True and folder.is_dir() is True:
                model_folders.append(folder)

        # sort folders by date such that the most recent date is the first element
        for folder in sorted(model_folders, reverse=True):
            model_file = folder / self.MODEL_FILENAME
            if model_file.is_file() is True:
                model_folder = folder
                break
        # we did not find a valid model file
        else:
            msg = f"No model file found at save location: '{pid_model_folder}'"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        return model_file, model_folder

    def build_save_folder_path(self, pid):

        pid_model_folder = self.build_pid_model_folder_path(pid=pid)

        datetime_str = datetime.utcnow().strftime(self.FOLDER_DATETIME_FORMAT)

        save_folder = pid_model_folder / datetime_str

        return save_folder

    @classmethod
    def determine_model_age_from_path(cls, model_location):
        if model_location is None:
            return float("inf")

        # trained_models/<pid>/<YYYYMMDDHHMMSS>/model.bin
        datetime_string = model_location.parent.name
        model_datetime = datetime.strptime(datetime_string, cls.FOLDER_DATETIME_FORMAT)
        model_age = datetime.utcnow() - model_datetime

        model_age_days = model_age.days

        return model_age_days
