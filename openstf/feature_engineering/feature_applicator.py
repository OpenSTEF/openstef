from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from openstf.feature_engineering.apply_features import apply_features

LATENCY_CONFIG = {"APX": 24} # A specific latency is part of a specific feature.

class AbstractFeatureApplicator(ABC):

    def __init__(self, feature_set_list, horizons):
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
            res = apply_features(df.copy(), h_ahead=horizon)
            res["Horizon"] = horizon
            if len(res.columns) > len(cols):
                cols = res.columns
            result = result.append(res, sort=False)  # appending unsorts columns

        # apply data latency
        for feature, time in LATENCY_CONFIG.items():
            result.loc[result["Horizon"] > time, feature] = np.nan

        return result[cols].sort_index()


class PredictFeatureApplicator(AbstractFeatureApplicator):

    def add_features(self, df):

        return apply_features(df, h_ahead=0.25)


class BackTestFeatureApplicator(AbstractFeatureApplicator):

    def add_features(self, df):

        return apply_features(df)

