# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from abc import ABC, abstractmethod
from turtle import st
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.feature_engineering.apply_features import apply_features
from openstef.feature_engineering.feature_adder import (
    FeatureDispatcher,
    adders_from_modules,
)
from openstef.feature_engineering.general import (
    add_missing_feature_columns,
    enforce_feature_order,
    remove_non_requested_feature_columns,
)

LATENCY_CONFIG = {"APX": 24}  # A specific latency is part of a specific feature.


class AbstractFeatureApplicator(ABC):
    def __init__(
        self,
        horizons: Union[List[float], str],
        feature_names: Optional[List[str]] = None,
        feature_modules: Optional[List[str]] = [],
    ) -> None:
        """Initialize abstract feature applicator.

        Args:
            horizons (list): list of horizons in hours
            feature_names (List[str]):  List of requested features
        """
        if not isinstance(horizons, str) and type(horizons) is not list and not None:
            raise ValueError("horizons must be added as a list")

        self.feature_names = feature_names
        self.horizons = horizons
        self.features_adder = adders_from_modules(feature_modules)
        self.features_dispatcher = FeatureDispatcher(self.features_adder)

    @abstractmethod
    def add_features(
        self, df: pd.DataFrame, pj: PredictionJobDataClass = None
    ) -> pd.DataFrame:
        """Adds features to an input DataFrame

        Args:
            df: pd.DataFrame with input data to which the features have to be added
            pj: (Optional) A prediction job that is needed for location dependent features,
                if not specified a default location is used
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
            pj: (Optional) A prediction job that is needed for location dependent features,
                if not specified a default location is used
            latency_config (dict): Optional. Invalidate certain features that are not
                available for a specific horizon due to data latency. Default to
                {"APX": 24}

        Returns:
            pd.DataFrame: Input DataFrame with an extra column for every added feature
                and sorted on the datetime index.
        """

        # If pj is none add empty dict
        if pj is None:
            pj = {}

        if latency_config is None:
            latency_config = LATENCY_CONFIG

        # Set default horizons if none are provided
        if self.horizons is None:
            self.horizons = [0.25, 24]

        # Pre define output variables
        result = pd.DataFrame()

        if isinstance(self.horizons, str):
            # copy the custom horizon into the horizon column
            res = df.copy(deep=True)
            res["horizon"] = res[self.horizons]
            result = pd.concat([result, res])
        else:
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
                result = pd.concat([result, res])

        # Add custom features with the dispatcher
        result = self.features_dispatcher.apply_features(
            result, feature_names=self.feature_names
        )

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
            if pj.get("model") == "proloaf":
                features = self.feature_names + ["historic_load"] + ["horizon"]
            else:
                features = self.feature_names + ["horizon"]
            result = remove_non_requested_feature_columns(result, features)

        # Sort all features except for the (first) load and (last) horizon columns
        return enforce_feature_order(result)


class OperationalPredictFeatureApplicator(AbstractFeatureApplicator):
    def add_features(
        self, df: pd.DataFrame, pj: PredictionJobDataClass = None
    ) -> pd.DataFrame:
        """Adds features to an input DataFrame.

        This method is implemented specifically for an operational prediction pipeline
         and will add every available feature.

        Args:
            df: pd.DataFrame with input data to which the features have to be added
            pj: (Optional) A prediction job that is needed for location dependent features,
                if not specified a default location is used
        Returns:
            pd.DataFrame: Input DataFrame with an extra column for every added feature.

        """

        # If pj is none add empty dict
        if pj is None:
            pj = {}

        num_horizons = len(self.horizons)
        if num_horizons != 1:
            raise ValueError(f"Expected one horizon, got {num_horizons}")

        # Add core features
        df = apply_features(
            df, feature_names=self.feature_names, horizon=self.horizons[0], pj=pj
        )
        # Add custom features with the dispatcher
        df = self.features_dispatcher.apply_features(
            df, feature_names=self.feature_names
        )

        df = add_missing_feature_columns(df, self.feature_names)

        # NOTE this is required since apply_features could add additional features
        if self.feature_names is not None:
            df = remove_non_requested_feature_columns(df, self.feature_names)

        return enforce_feature_order(df)
