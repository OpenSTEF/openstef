import importlib
import structlog

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from datetime import timedelta

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.feature_engineering.feature_applicator import (
    TrainFeatureApplicator,
    OperationalPredictFeatureApplicator,
)
from openstef.feature_engineering.general import (
    enforce_feature_order,
    remove_non_requested_feature_columns,
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

    def check_model(self):
        if self.model is None:
            raise ValueError(
                "If no model has been provided to the feature builder , it cannot perform processing for forecast task!"
            )


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
        self.check_model()

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


class ARFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        pj: PredictionJobDataClass,
        model_specs: ModelSpecificationDataClass,
        model: Optional[OpenstfRegressor] = None,
        horizons: Optional[list[float]] = None,
        historical_depth: Optional[int] = None,
    ) -> None:
        super().__init__(pj, model_specs, model, horizons)
        self.historical_depth = historical_depth

    def process_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add dummy horizon column
        data["horizon"] = 0
        # data.index.freq = f"{self.pj.resolution_minutes}min"
        features = self.model_specs.feature_names + ["horizon"]
        result = remove_non_requested_feature_columns(data, features)

        # Sort all features except for the (first) load and (last) horizon columns
        return enforce_feature_order(result)

    def process_forecast_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger = structlog.get_logger(__name__)
        self.check_model()
        # Prep forecast input by selecting only the forecast datetime interval (this is much smaller than the input range)
        # Also drop the load column
        forecast_start, forecast_end = generate_forecast_datetime_range(data)
        forecast_input_data = data[forecast_start:forecast_end].drop(columns="load")

        historical_start = None
        if self.historical_depth:
            historical_start = forecast_start - self.historical_depth * timedelta.min(
                self.pj.resolution_minutes
            )
        past_data = data[historical_start:forecast_start]
        self.model.update_historic(past_data["load"], past_data.drop(columns="load"))
        logger.info(
            "Watch-out side effect on the model performed in the feature builder to update the historical data."
        )

        return forecast_input_data, data
