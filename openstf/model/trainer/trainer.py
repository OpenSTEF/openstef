# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from openstf.enums import MLModelType
from openstf.model.serializer.creator import ModelSerializerCreator


class AbstractModelTrainer(ABC):
    def __init__(self, pj):
        super().__init__()
        self.pj = pj
        self.trained_model = None
        self.old_model = None
        self.confidence_interval = None
        self.figs = None
        self.hyper_parameters = None

        self.serializer = ModelSerializerCreator().create_model_serializer(
            MLModelType(self.pj["model"])
        )
        self.old_model, old_model_location = self.serializer.load(self.pj["id"])
        self.old_model_age = self.serializer.determine_model_age_from_path(
            old_model_location
        )

    @property
    @abstractmethod
    def feature_importance(self):
        """Abstract method that gets feature importances of trained model.

        Returns:
            pandas.DataFrame: A DataFrame describing the feature importances of
            the trained model. At least the following columns should be
            included::

            index (str): Label of feature
            importance (float): Indication of how important the feature was to
            the trained model. Use only non-negative values. Does not need to
            sum to 1.
            weight (float): Indication of how often the feature was used in the
            trained model. Use only non-negative values. Should be normalized to
            sum to 1.

            If the model only supports a single importance/weight indication,
            i.e linear regression, set both columns to this value.

        """
        print(
            "This is an abstract class please use a derivative class where a \
            get_feature_importance method is implemented."
        )

    @abstractmethod
    def train(self, train_data, validation_data, callbacks=None):
        """Abstract method to train a model based on train and validation data.

        The method should add the newly trained model to the trained_model
        attribute.

        Args:
            train_data (pandas.DataFrame): The train data
            validation_data (pandas.DataFrame): The validation data
            callbacks (list of callable): List of callback functions that can be
                called at the end of each training iteration

        """
        print(
            "This is an abstract class please use a derivative class where a \
            train method is implemented."
        )

    @abstractmethod
    def better_than_old_model(self, test_data):
        """Abstract method that checks if new model is better than the old model.

        Args:
            test_data (pandas.DataFrame): The test data

        Returns:
            Bool: True if new model is better than old model, false otherwise

        """
        print(
            "This is an abstract class please use a derivative class where a \
            better_than_old_model method is implemented."
        )
        return

    @abstractmethod
    def store_model(self):
        """Abstract method that should store the model.

        The general implementation is to just call _store_model. However this
        function exists if some other steps must be done before _store_model can
        be called.

        """
        print(
            "This is an abstract class please use a derivative class where a \
            store_model method is implemented."
        )

    @abstractmethod
    def hyper_params_objective(
        self,
        trial,
        error_function=None,
        clean_data_with_features=None,
        featuresets=None,
        optimize_peak_prediction=False,
    ):
        print(
            "This is an abstract class please use a derivative class where a \
            get_corrections method is implemented."
        )

    def _store_model(self):
        """
            Method that stores a newly trained model to persistent storage.

        Args:
            model_type: ModelType enum that specifies to look for a component model
            name: str() name of the component
        """
        self.serializer.save(
            self.pj["id"], self.trained_model, corrections=self.confidence_interval
        )

    @staticmethod
    def _calculate_confidence_interval(realised, predicted):
        """Protected static method to calculate the corrections for a model

        Args:
            realised: pd.series with realised load
            predicted: pd.series with load predicted by new model

        Returns:
            pd.DataFrame: with model corrections
        """
        result = pd.DataFrame(index=range(24), columns=["stdev", "hour"])
        # Calculate the error for each predicted point
        error = realised - predicted
        error.index = error.index.hour  # Hour only, remove the rest
        # For the time starts with 00, 01, 02, etc. TODO (MAKE MORE ELEGANT SOLUTION THAN A LOOP)
        for hour in range(24):
            hour_error = error[error.index == hour]
            result["stdev"].iloc[hour] = np.std(hour_error)
            result["hour"].iloc[hour] = hour

        result = result.astype("float")

        return result
