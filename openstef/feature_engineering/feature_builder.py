import importlib

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.feature_engineering.feature_applicator import (
    TrainFeatureApplicator,
    OperationalPredictFeatureApplicator,
)
from openstef.pipeline.utils import generate_forecast_datetime_range


def dynamic_load(name):
    components = name.split(".")
    module = importlib.import_module(".".join(components[:-1]))
    my_class = getattr(module, components[-1])
    return my_class


class AbstractFeatureBuilder(ABC):
    def __init__(
        self,
        pj: PredictionJobDataClass,
        model_specs: ModelSpecificationDataClass,
        model: Optional[OpenstfRegressor] = None,
        horizons: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.pj = pj
        self.model_specs = model_specs
        self.model = model
        self.horizons = horizons

    @abstractmethod
    def process_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def process_forecast_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass


class LegacyFeatureBuilder(AbstractFeatureBuilder):
    def process_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.horizons:
            horizons = self.horizons
        else:
            horizons = self.pj.resolution_minutes

        features_applicator = TrainFeatureApplicator(
            horizons=horizons,
            feature_names=self.model_specs.feature_names,
            feature_modules=self.model_specs.feature_modules,
        )
        return features_applicator.add_features(data, pj=self.pj)

    def process_forecast_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError(
                "If no model has been provided to the feature builder , it cannot perform processing for forecast task!"
            )
        features_applicator = OperationalPredictFeatureApplicator(
            horizons=[self.pj["resolution_minutes"] / 60.0],
            feature_names=self.model.feature_names,
            feature_modules=self.model_specs.feature_modules,
        )
        data_with_features = features_applicator.add_features(data)

        # Prep forecast input by selecting only the forecast datetime interval (this is much smaller than the input range)
        # Also drop the load column
        forecast_start, forecast_end = generate_forecast_datetime_range(
            data_with_features
        )
        forecast_input_data = data_with_features[forecast_start:forecast_end].drop(
            columns="load"
        )

        return forecast_input_data, data_with_features
