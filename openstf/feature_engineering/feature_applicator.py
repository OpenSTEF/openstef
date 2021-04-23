# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from openstf.feature_engineering.apply_features import apply_features
from openstf.feature_engineering.general import (
    add_missing_feature_columns,
    remove_extra_feature_columns,
)

LATENCY_CONFIG = {"APX": 24}  # A specific latency is part of a specific feature.


class AbstractFeatureApplicator(ABC):
    def __init__(self, horizons: list, feature_set_list: list = None) -> None:
        """

        Args:
            horizons: (list) list of horizons
            feature_set_list: (list) List of requested features
        """
        if type(horizons) is not list and not None:
            raise ValueError("Horizons must be added as a list")

        self.feature_set_list = feature_set_list
        self.horizons = horizons

    @abstractmethod
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Adds features to an input DataFrame

        Args:
            df: pd.DataFrame with input data to which the features have to be added
        """
        pass


class TrainFeatureApplicator(AbstractFeatureApplicator):
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Adds features to an input DataFrame.
        This method is implemented specifically for a model train pipeline.
        For larger horzions data is invalidated as when they are not available.
        For example:
            For horzion 24 hours the feature T-720min is not added as the load 720 minutes ago is not available 24 hours in advance.
            In case of a horizon 0.25 hours this feature is added as in this case the feature is available.

        Args:
            df: pd.DataFrame with input data to which the features have to be added

        Returns:
            pd.DataFrame: Input DataFrame with an extra column for every added feature.

        """

        # Set default horizons if none are provided
        if self.horizons is None:
            self.horizons = [0.25, 24]

        # Pre define output vairables
        result = pd.DataFrame()
        cols = []

        # Loop over horizons and add corresponding features
        for horizon in self.horizons:
            res = apply_features(df.copy(), horizon=horizon)
            res["Horizon"] = horizon
            if len(res.columns) > len(cols):
                cols = res.columns
            result = result.append(res, sort=False)  # appending unsorts columns

        # Invalidate features that are not available for a specific horizon due to data latency
        for feature, time in LATENCY_CONFIG.items():
            result.loc[result["Horizon"] > time, feature] = np.nan

        return result[cols].sort_index()


class OperationalPredictFeatureApplicator(AbstractFeatureApplicator):
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds features to an input DataFrame.
        This method is implemented specifically for an operational prediction pipeline
         and will add every available feature.

        Args:
            df: pd.DataFrame with input data to which the features have to be added

        Returns:
            pd.DataFrame: Input DataFrame with an extra column for every added feature.

        """
        if self.horizons is None:
            self.horizons = [0.25]

        df = apply_features(
            df, feature_set_list=self.feature_set_list, horizon=self.horizons[0]
        )
        df = add_missing_feature_columns(df, self.feature_set_list)
        df = remove_extra_feature_columns(df, self.feature_set_list)

        return df


class BackTestPredictFeatureApplicator(AbstractFeatureApplicator):
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Adds features to an input DataFrame.
        This method is implemented specifically for a backtest prediction for a specific horizon.
        All featurs that are not available for the specific horzion are invalidated.

        Args:
            df: pd.DataFrame with input data to which the features have to be added

        Returns:
            pd.DataFrame: Input DataFrame with an extra column for every added feature.
        """
        if self.horizons is None:
            self.horizons = [24.0]

        if len(self.horizons) > 1:
            raise ValueError("Prediction can only be done one horizon at a time!")

        df = apply_features(df, horizon=self.horizons[0])
        df = add_missing_feature_columns(df, self.feature_set_list)
        df = remove_extra_feature_columns(df, self.feature_set_list)
        return df
