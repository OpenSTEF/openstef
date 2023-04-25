# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
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


class AbstractDataPreparation(ABC):
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
    def prepare_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def prepare_forecast_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def check_model(self):
        if self.model is None:
            raise ValueError(
                "If no model has been provided to the data prep class, it cannot perform preparation for forecast task!"
            )


class LegacyDataPreparation(AbstractDataPreparation):
    def prepare_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def prepare_forecast_data(self, data: pd.DataFrame) -> pd.DataFrame:
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


class ARDataPreparation(AbstractDataPreparation):
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

    def prepare_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add dummy horizon column
        data["horizon"] = 0

        # remove non requested feature
        features = self.model_specs.feature_names + ["horizon"]
        result = remove_non_requested_feature_columns(data, features)

        # Sort all features except for the (first) load and (last) horizon columns
        result = result[["load"] + [c for c in result.columns if c != "load"]]
        result = result.sort_index()
        result = enforce_feature_order(result)

        result = result[result.iloc[:, 0].notna()]
        return result

    def prepare_forecast_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger = structlog.get_logger(__name__)
        self.check_model()
        # Prep forecast input by selecting only the forecast datetime interval (this is much smaller than the input range)
        # Also drop the load column
        data = data[["load"] + self.model.feature_names]
        forecast_start, forecast_end = generate_forecast_datetime_range(data)
        forecast_input_data = data[forecast_start:forecast_end].drop(columns="load")

        historical_start = None
        if self.historical_depth:
            historical_start = forecast_start - self.historical_depth * timedelta(
                minutes=self.pj.resolution_minutes
            )
        past_data = data[historical_start:forecast_start].iloc[:-1]
        self.model.update_historic_data(
            past_data.drop(columns="load"), past_data["load"]
        )
        logger.info(
            "Watch-out side effect on the model performed in the feature builder to update the historical data."
        )

        data[self.model.feature_importance_dataframe.index.tolist()] = 0
        return forecast_input_data, data
