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
    def __init__(self, horizons, feature_set_list = None):

        if horizons is not list and not None:
            raise ValueError("Horizons must be added as a list")

        self.feature_set_list = feature_set_list
        self.horizons = horizons

    @abstractmethod
    def add_features(self, df):
        pass


class TrainFeatureApplicator(AbstractFeatureApplicator):
    def add_features(self, df):
        if self.horizons is None:
            self.hozizons = [0.25, 24]

        result = pd.DataFrame()
        cols = []
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

    def add_features(self, df):
        if self.horizons is None:
            self.horizons = [0.25]

        df = apply_features(df, feature_set_list=self.feature_set_list, horizon=self.horizons[0])
        df = add_missing_feature_columns(df, self.feature_set_list)
        df = remove_extra_feature_columns(df, self.feature_set_list)

        return df


class BackTestPredictFeatureApplicator(AbstractFeatureApplicator):

    def add_features(self, df):
        if self.horizons is None:
            self.horizons = [24.0]

        if len(self.horizons) > 1:
            raise ValueError("Prediction can only be done one horizon at a time!")

        df = apply_features(df, horizon=self.horizons[0])
        df = add_missing_feature_columns(df, self.feature_set_list)
        df = remove_extra_feature_columns(df, self.feature_set_list)
        return df
