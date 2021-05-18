# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import joblib
import structlog
from ktpbase.config.config import ConfigManager
from sklearn.base import RegressorMixin

MODEL_FILENAME = "model.joblib"
FOLDER_DATETIME_FORMAT = "%Y%m%d%H%M%S"


class AbstractSerializer(ABC):
    def __init__(self, prediction_job: dict) -> None:
        self.pj = prediction_job
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.trained_models_folder = Path(ConfigManager.get_instance().paths.trained_models)

    @abstractmethod
    def save_model(self, model: RegressorMixin) -> None:
        """Persists trained sklearn compantible model

        Args:
            model: Trained sklearn compatible model object
        """
        self.logger.error("This is an abstract method!")
        pass

    @abstractmethod
    def load_model(self) -> RegressorMixin:
        """Loads model that has been trained earlier

        Returns: Trained sklearn compatible model object

        """
        self.logger.error("This is an abstract method!")


class PersistentStorageSerializer(AbstractSerializer):

    def save_model(self, model: RegressorMixin) -> None:
        """Serializes trained sklearn compantible model to persistent storage

        Args:
            model: Trained sklearn compatible model object
        """

        # Compose save path
        save_path = self._build_save_folder_path()

        # Create save path if nescesarry
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, save_path / MODEL_FILENAME)

    def load_model(self) -> RegressorMixin:
        """Loads model from persistent storage that has been trained earlier

        Raises:
            FileNotFoundError: When no model file is found in the give directory.

        Returns: Trained sklearn compatible model object

        """

        # Get the folder specific to this prediction job
        pid_model_folder = self._build_pid_model_folder_path()

        # Get the most recent model folder
        location_most_recent_model = self._find_most_recent_model_folder(
            pid_model_folder
        )

        # Load most recent model from disk
        try:
            loaded_model = joblib.load(location_most_recent_model / MODEL_FILENAME)
        except Exception as e:
            self.logger.error(
                "Could not load most recent model!", pid=self.pj["id"], exception=e
            )
            raise FileNotFoundError("Could not load model from the model file!")

        # exctract model age
        model_age_in_days = float("inf")  # In case no model is loaded,
        # we still need to provide an age
        if loaded_model is not None:
            model_age_in_days = self._determine_model_age_from_path(
                location_most_recent_model
            )

        # Add model age to model object
        loaded_model.age = model_age_in_days

        return loaded_model

    def _build_pid_model_folder_path(self) -> Path:
        """Build the trained models path for the given pid.
        The trainded models are stored a folder structure using the following
        template: <trained_models_folder>/<pid>[_<component-name>]/<YYYYMMDDHHMMSS>/

        """

        # Folder name is equal to pid
        model_folder_name = f"{self.pj['id']}"

        # Use custom folder if specified otherwise use the one from the config manager
        trained_models_folder = self.trained_models_folder

        # Combine into complete folder path
        model_folder = trained_models_folder / model_folder_name

        return model_folder

    def _find_most_recent_model_folder(self, pid_model_folder: Path) -> Path:
        """Find the model recent model folder.
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

        # Declare empty list to append folder names
        model_folders = []

        # Loop over all subfolder of pid_model_folder and store the ones starting with
        # a 2 (date starts with 2000s), this is nesscerary for legacy reasons.
        for folder in pid_model_folder.iterdir():
            if folder.name.startswith("2") is True and folder.is_dir() is True:
                model_folders.append(folder)

        # sort folders by date such that the most recent date is the first element
        for folder in sorted(model_folders, reverse=True):
            model_file = folder / MODEL_FILENAME
            if model_file.is_file() is True:
                model_folder = folder
                break
        # we did not find a valid model file
        else:
            msg = f"No model file found at save location: '{pid_model_folder}'"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        return model_folder

    def _build_save_folder_path(self) -> Path:
        """Builds a save path where to save a model.

        Returns: pathlib.Path() with location where to save a model

        """

        # Get specific folder for this precion job
        pid_model_folder = self._build_pid_model_folder_path()

        # Get the datetime string
        datetime_str = datetime.utcnow().strftime(FOLDER_DATETIME_FORMAT)

        # Combine into a complete model save path
        save_folder = pid_model_folder / datetime_str

        return save_folder

    def _determine_model_age_from_path(self, model_location: Path) -> float:
        """Determines how many days ago a model is trained base on the folder name.

        Args:
            model_location: pathlib.Path: Path to the model folder

        Returns: Number of days since training of the model

        """

        # Location is of this format: TRAINED_MODELS_FOLDER/<pid>/<YYYYMMDDHHMMSS>/
        datetime_string = model_location.name

        # Convert string to datetime object
        try:
            model_datetime = datetime.strptime(datetime_string, FOLDER_DATETIME_FORMAT)
        except Exception as e:
            self.logger.warning(
                "Could not parse model folder name to determine model age. Returning infinite age!",
                exception=e,
                folder_name=datetime_string,
            )
            return float("inf")  # Return fallback age

        # Get time difference between now and training in days
        model_age_days = (datetime.utcnow() - model_datetime).days

        return model_age_days
