# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd

from openstef.feature_engineering.apply_features import apply_features
from openstef.feature_engineering.general import (
    add_missing_feature_columns,
    enforce_feature_order,
    remove_non_requested_feature_columns,
)
from openstef_dbc.services.prediction_job import PredictionJobDataClass

LATENCY_CONFIG = {"APX": 24}  # A specific latency is part of a specific feature.


class AbstractFeatureApplicator(ABC):
    def __init__(
        self, horizons: List[float], feature_names: Optional[List[str]] = None
    ) -> None:
        """Initialize abstract feature applicator.

        Args:
            horizons (list): list of horizons in hours
            feature_names (List[str]):  List of requested features
        """
        if type(horizons) is not list and not None:
            raise ValueError("horizons must be added as a list")

        self.feature_names = feature_names
        self.horizons = horizons

    @abstractmethod
    def add_features(
        self, df: pd.DataFrame, pj: PredictionJobDataClass
    ) -> pd.DataFrame:
        """Adds features to an input DataFrame

        Args:
            df: pd.DataFrame with input data to which the features have to be added
            pj (PredictionJobDataClass): Prediction job.
        """
        pass


class TrainFeatureApplicator(AbstractFeatureApplicator):
    def add_features(
        self, df: pd.DataFrame, pj: PredictionJobDataClass = None, latency_config=None
    ) -> pd.DataFrame:
        """Adds features to an input DataFrame.

        This method is implemented specifically for a model train pipeline. For larger
        horzions data is invalidated as when they are not available.

        For example:
            For horzion 24 hours the feature T-720min is not added as the load
            720 minutes ago is not available 24 hours in advance. In case of a horizon
            0.25 hours this feature is added as in this case the feature is available.

        Args:
            df (pd.DataFrame):  Input data to which the features will be added.
            pj (PredictionJobDataClass): Prediction job.
            latency_config (dict): Optional. Invalidate certain features that are not
                available for a specific horizon due to data latency. Default to
                {"APX": 24}

        Returns:
            pd.DataFrame: Input DataFrame with an extra column for every added feature
                and sorted on the datetime index.
        """

        if latency_config is None:
            latency_config = LATENCY_CONFIG

        # Set default horizons if none are provided
        if self.horizons is None:
            self.horizons = [0.25, 24]

        # Pre define output variables
        result = pd.DataFrame()

        # Loop over horizons and add corresponding features
        for horizon in self.horizons:
            # Deep copy of df is important, because we want a fresh start every iteration!
            res = apply_features(
                df.copy(deep=True),
                horizon=horizon,
                pj=pj,
                feature_names=self.feature_names,
            )
            res["horizon"] = horizon
            result = result.append(res)

        # IMPORTANT: sort index to prevent errors when slicing on the (datetime) index
        # if we don't sort, the duplicated indexes (one per horizon) have large gaps
        # and slicing will give an exception.
        result = result.sort_index(axis=0)

        # Invalidate features that are not available for a specific horizon due to data
        # latency
        for feature, time in latency_config.items():
            result.loc[result["horizon"] > time, feature] = np.nan

        # NOTE this is required since apply_features could add additional features
        if self.feature_names is not None:
            # Add horizon to requested features else it is removed, and if needed the proloaf feature (historic_load)
            if pj is not None and pj["model"] == "proloaf":
                features = self.feature_names + ["historic_load"] + ["horizon"]
            else:
                features = self.feature_names + ["horizon"]
            result = remove_non_requested_feature_columns(result, features)

        # Sort all features except for the (first) load and (last) horizon columns
        return enforce_feature_order(result)


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
        num_horizons = len(self.horizons)
        if num_horizons != 1:
            raise ValueError(f"Expected one horizon, got {num_horizons}")

        df = apply_features(
            df, feature_names=self.feature_names, horizon=self.horizons[0]
        )

        df = add_missing_feature_columns(df, self.feature_names)

        # NOTE this is required since apply_features could add additional features
        if self.feature_names is not None:
            df = remove_non_requested_feature_columns(df, self.feature_names)

        return enforce_feature_order(df)
