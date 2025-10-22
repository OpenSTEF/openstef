# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from typing import Literal

from pydantic import Field
from pydantic_extra_types.coordinate import Coordinate, Latitude, Longitude
from pydantic_extra_types.country import CountryAlpha2

from openstef_beam.evaluation.metric_providers import (
    MetricDirection,
    MetricProvider,
    ObservedProbabilityProvider,
    R2Provider,
)
from openstef_core.base_model import BaseConfig
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q, Quantile, QuantileOrGlobal
from openstef_models.integrations.mlflow import MLFlowStorageCallback
from openstef_models.mixins import ModelIdentifier
from openstef_models.models import ForecastingModel
from openstef_models.models.forecasting.flatliner_forecaster import FlatlinerForecaster
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearForecaster
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster
from openstef_models.transforms.energy_domain import WindPowerFeatureAdder
from openstef_models.transforms.general import Clipper, EmptyFeatureRemover, Imputer
from openstef_models.transforms.postprocessing import ConfidenceIntervalApplicator, QuantileSorter
from openstef_models.transforms.time_domain import (
    CyclicFeaturesAdder,
    DatetimeFeaturesAdder,
    HolidayFeatureAdder,
    RollingAggregatesAdder,
)
from openstef_models.transforms.time_domain.rolling_aggregates_adder import AggregationFunction
from openstef_models.transforms.validation import CompletenessChecker, FlatlineChecker
from openstef_models.transforms.weather_domain import DaylightFeatureAdder, RadiationDerivedFeaturesAdder
from openstef_models.transforms.weather_domain.atmosphere_derived_features_adder import AtmosphereDerivedFeaturesAdder
from openstef_models.utils.feature_selection import Exclude, FeatureSelection, Include
from openstef_models.workflows import CustomForecastingWorkflow


class ForecastingWorkflowConfig(BaseConfig):
    model_id: ModelIdentifier = Field(default=...)

    # Model configuration
    model: Literal["xgboost", "gblinear", "flatliner"] = Field(default=...)  # TODO(#652): Implement median forecaster
    hyperparams: XGBoostForecaster.HyperParams | GBLinearForecaster.HyperParams = Field(default=...)
    quantiles: list[Quantile] = Field(default=[Q(0.5)])

    sample_interval: timedelta = Field(default=timedelta(minutes=15))
    horizons: list[LeadTime] = Field(default=[LeadTime.from_string("PT48H")])

    xgboost_hyperparams: XGBoostForecaster.HyperParams = Field(default=XGBoostForecaster.HyperParams())
    gblinear_hyperparams: GBLinearForecaster.HyperParams = Field(default=GBLinearForecaster.HyperParams())

    # Location information
    name: str = Field(default="test_location")
    description: str = Field(default="")
    coordinate: Coordinate = Field(
        default=Coordinate(
            latitude=Latitude(52.132633),
            longitude=Longitude(5.291266),
        )
    )
    country_code: CountryAlpha2 = Field(default=CountryAlpha2("NL"))

    # Data properties
    target_column: str = Field(default="load")
    energy_price_column: str = Field(default="day_ahead_electricity_price")

    # Feature engineering and validation
    completeness_threshold: float = Field(
        default=0.5, description="Minimum fraction of data that should be available for making a regular forecast."
    )
    flatliner_threshold: timedelta = Field(
        default=timedelta(hours=24),
        description="Number of minutes that the load has to be constant to detect a flatliner.",
    )
    detect_non_zero_flatliner: bool = Field(
        default=False,
        description="If True, flatliners are also detected on non-zero values (median of the load).",
    )
    rolling_aggregate_features: list[AggregationFunction] = Field(
        default=[],
        description="If not None, rolling aggregate(s) of load will be used as features in the model.",
    )
    clip_features: FeatureSelection = Field(
        default=FeatureSelection(include=None, exclude=None),
        description="Feature selection for which features to clip.",
    )

    # Data splitting strategy
    val_fraction: float = Field(
        default=0.15,
        description="Fraction of data to reserve for the validation set when automatic splitting is used.",
    )
    test_fraction: float = Field(
        default=0.1,
        description="Fraction of data to reserve for the test set when automatic splitting is used.",
    )
    stratification_fraction: float = Field(
        default=0.15,
        description="Fraction of extreme values to use for stratified splitting into train/test sets.",
    )
    min_days_for_stratification: int = Field(
        default=4,
        description="Minimum number of unique days required to perform stratified splitting.",
    )

    # Evaluation
    evaluation_metrics: list[MetricProvider] = Field(
        default_factory=lambda: [R2Provider(), ObservedProbabilityProvider()],
        description="List of metric providers for evaluating model score.",
    )

    # Callbacks
    model_reuse_enable: bool = Field(default=True)
    model_reuse_max_age: timedelta = Field(default=timedelta(days=7))

    model_selection_enable: bool = Field(default=True)
    model_selection_metric: tuple[QuantileOrGlobal, str, MetricDirection] = Field(
        default=(Q(0.5), "R2", "higher_is_better"),
        description="Metric to monitor for model performance when retraining.",
    )
    model_selection_old_model_penalty: float = Field(
        default=1.2,
        description="Penalty to apply to the old model's metric to bias selection towards newer models.",
    )

    # Metadata
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for the model.",
    )


def create_forecasting_workflow(config: ForecastingWorkflowConfig) -> CustomForecastingWorkflow:
    checks = [
        FlatlineChecker(
            load_column=config.target_column,
            flatliner_threshold=config.flatliner_threshold,
            detect_non_zero_flatliner=config.detect_non_zero_flatliner,
            error_on_flatliner=True,
        ),
        CompletenessChecker(completeness_threshold=config.completeness_threshold),
    ]
    feature_adders = [
        WindPowerFeatureAdder(),
        AtmosphereDerivedFeaturesAdder(),
        RadiationDerivedFeaturesAdder(),
        CyclicFeaturesAdder(),
        DaylightFeatureAdder(),
        RollingAggregatesAdder(
            feature=config.target_column,
            aggregation_functions=config.rolling_aggregate_features,
        ),
        # TODO: Add normal horizon based lags.
    ]
    feature_standardizers = [
        Clipper(selection=Include(config.energy_price_column).combine(config.clip_features), mode="standard"),
        EmptyFeatureRemover(),
    ]

    if config.model == "xgboost":
        preprocessing = [
            *checks,
            *feature_adders,
            HolidayFeatureAdder(country_code=config.country_code),
            DatetimeFeaturesAdder(onehot_encode=False),
            *feature_standardizers,
        ]
        forecaster = XGBoostForecaster(
            config=XGBoostForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.xgboost_hyperparams,
            )
        )
        postprocessing = [QuantileSorter()]

    elif config.model == "gblinear":
        preprocessing = [
            *checks,
            Imputer(selection=Exclude(config.target_column), imputation_strategy="mean"),
            *feature_adders,
            *feature_standardizers,
        ]
        forecaster = GBLinearForecaster(
            config=GBLinearForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.gblinear_hyperparams,
            )
        )
        postprocessing = []
    elif config.model == "flatliner":
        preprocessing = []
        forecaster = FlatlinerForecaster(
            config=FlatlinerForecaster.Config(
                quantiles=[Q(0.5)],
                horizons=config.horizons,
            )
        )
        postprocessing = [
            ConfidenceIntervalApplicator(quantiles=config.quantiles),
        ]
    else:
        msg = f"Unsupported model type: {config.model}"
        raise ValueError(msg)

    tags = {
        "workflow_name": config.name,
        "workflow_description": config.description,
        "locatioin": str(config.coordinate),
        "country_code": str(config.country_code),
        "model_type": config.model,
        **config.tags,
    }

    return CustomForecastingWorkflow(
        model=ForecastingModel(
            preprocessing=TransformPipeline(transforms=preprocessing),
            forecaster=forecaster,
            postprocessing=TransformPipeline(transforms=postprocessing),
            target_column=config.target_column,
            # Data splitting strategy
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
            stratification_fraction=config.stratification_fraction,
            min_days_for_stratification=config.min_days_for_stratification,
            # Evaluation
            evaluation_metrics=config.evaluation_metrics,
            # Other
            tags=tags,
        ),
        model_id=config.model_id,
        callbacks=[
            MLFlowStorageCallback(
                model_reuse_enable=config.model_reuse_enable,
                model_reuse_max_age=config.model_reuse_max_age,
                model_selection_enable=config.model_selection_enable,
                model_selection_metric=config.model_selection_metric,
                model_selection_old_model_penalty=config.model_selection_old_model_penalty,
            )
        ],
    )
