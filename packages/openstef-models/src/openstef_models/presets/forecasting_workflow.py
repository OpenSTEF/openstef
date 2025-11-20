# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecasting workflow presets and configurations.

Provides predefined configurations and factory functions for common forecasting workflows,
including XGBoost, GBLinear, and Flatliner models with appropriate preprocessing pipelines.
"""

from datetime import timedelta
from decimal import Decimal
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
from openstef_models.integrations.mlflow import MLFlowStorage, MLFlowStorageCallback
from openstef_models.mixins import ModelIdentifier
from openstef_models.models import ForecastingModel
from openstef_models.models.forecasting.flatliner_forecaster import FlatlinerForecaster
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearForecaster
from openstef_models.models.forecasting.hybrid_forecaster import HybridForecaster
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster
from openstef_models.models.forecasting.lgbmlinear_forecaster import LGBMLinearForecaster
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster
from openstef_models.transforms.energy_domain import WindPowerFeatureAdder
from openstef_models.transforms.general import Clipper, EmptyFeatureRemover, Imputer, NaNDropper, SampleWeighter, Scaler
from openstef_models.transforms.postprocessing import ConfidenceIntervalApplicator, QuantileSorter
from openstef_models.transforms.time_domain import (
    CyclicFeaturesAdder,
    DatetimeFeaturesAdder,
    HolidayFeatureAdder,
    RollingAggregatesAdder,
)
from openstef_models.transforms.time_domain.lags_adder import LagsAdder
from openstef_models.transforms.time_domain.rolling_aggregates_adder import AggregationFunction
from openstef_models.transforms.validation import CompletenessChecker, FlatlineChecker, InputConsistencyChecker
from openstef_models.transforms.weather_domain import (
    AtmosphereDerivedFeaturesAdder,
    DaylightFeatureAdder,
    RadiationDerivedFeaturesAdder,
)
from openstef_models.utils.data_split import DataSplitter
from openstef_models.utils.feature_selection import Exclude, FeatureSelection, Include
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)


class LocationConfig(BaseConfig):
    """Configuration for location information in forecasting workflows."""

    name: str = Field(
        default="test_location",
        description="Name of the forecasting location or workflow.",
    )
    description: str = Field(default="", description="Description of the forecasting workflow.")
    coordinate: Coordinate = Field(
        default=Coordinate(
            latitude=Latitude(Decimal("52.132633")),
            longitude=Longitude(Decimal("5.291266")),
        ),
        description="Geographic coordinate of the location.",
    )
    country_code: CountryAlpha2 = Field(
        default=CountryAlpha2("NL"),
        description="Country code for holiday feature generation.",
    )

    @property
    def tags(self) -> dict[str, str]:
        """Generate tags dictionary from location information."""
        return {
            "location_name": self.name,
            "location_description": self.description,
            "location_coordinate": str(self.coordinate),
            "location_country_code": str(self.country_code),
        }


class ForecastingWorkflowConfig(BaseConfig):  # PredictionJob
    """Configuration for forecasting workflows.

    Defines all parameters needed to set up a forecasting model, including model type,
    hyperparameters, location information, data columns, and feature engineering settings.
    """

    model_id: ModelIdentifier = Field(description="Unique identifier for the forecasting model.")

    # Model configuration
    model: Literal["xgboost", "gblinear", "flatliner", "hybrid", "lgbm", "lgbmlinear"] = Field(
        description="Type of forecasting model to use."
    )  # TODO(#652): Implement median forecaster
    quantiles: list[Quantile] = Field(
        default=[Q(0.5)],
        description="List of quantiles to predict for probabilistic forecasting.",
    )

    sample_interval: timedelta = Field(
        default=timedelta(minutes=15),
        description="Time interval between consecutive data samples.",
    )
    horizons: list[LeadTime] = Field(
        default=[LeadTime.from_string("PT48H")],
        description="List of forecast horizons to predict.",
    )

    xgboost_hyperparams: XGBoostForecaster.HyperParams = Field(
        default=XGBoostForecaster.HyperParams(),
        description="Hyperparameters for XGBoost forecaster.",
    )
    gblinear_hyperparams: GBLinearForecaster.HyperParams = Field(
        default=GBLinearForecaster.HyperParams(),
        description="Hyperparameters for GBLinear forecaster.",
    )

    lgbm_hyperparams: LGBMForecaster.HyperParams = Field(
        default=LGBMForecaster.HyperParams(),
        description="Hyperparameters for LightGBM forecaster.",
    )

    lgbmlinear_hyperparams: LGBMLinearForecaster.HyperParams = Field(
        default=LGBMLinearForecaster.HyperParams(),
        description="Hyperparameters for LightGBM forecaster.",
    )

    hybrid_hyperparams: HybridForecaster.HyperParams = Field(
        default=HybridForecaster.HyperParams(),
        description="Hyperparameters for Hybrid forecaster.",
    )

    location: LocationConfig = Field(
        default=LocationConfig(),
        description="Location information for the forecasting workflow.",
    )

    # Data properties
    target_column: str = Field(default="load", description="Name of the target variable column in datasets.")
    energy_price_column: str = Field(
        default="day_ahead_electricity_price",
        description="Name of the energy price column in datasets.",
    )
    radiation_column: str = Field(default="radiation", description="Name of the radiation column in datasets.")
    wind_speed_column: str = Field(default="windspeed", description="Name of the wind speed column in datasets.")
    pressure_column: str = Field(default="pressure", description="Name of the pressure column in datasets.")
    temperature_column: str = Field(default="temperature", description="Name of the temperature column in datasets.")
    relative_humidity_column: str = Field(
        default="relative_humidity",
        description="Name of the relative humidity column in datasets.",
    )
    predict_history: timedelta = Field(
        default=timedelta(days=14),
        description="Amount of historical data available at prediction time.",
    )
    cutoff_history: timedelta = Field(
        default=timedelta(days=0),
        description="Amount of historical data to exclude from training and prediction due to incomplete features "
        "from lag-based preprocessing. When using lag transforms (e.g., lag-14), the first N days contain NaN values. "
        "Set this to match your maximum lag duration (e.g., timedelta(days=14)). "
        "Default of 0 assumes no invalid rows are created by preprocessing. "
        "Note: should be same as predict_history if you are using lags. We default to disabled to keep the same "
        "behaviour as openstef 3.0.",
    )

    # Feature engineering and validation
    completeness_threshold: float = Field(
        default=0.5,
        description="Minimum fraction of data that should be available for making a regular forecast.",
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
    sample_weight_scale_percentile: int = Field(
        default=95,
        description="Percentile of target values used as scaling reference. "
        "Values are normalized relative to this percentile before weighting.",
    )
    sample_weight_exponent: float = Field(
        default_factory=lambda data: 1.0
        if data.get("model") in {"gblinear", "lgbmlinear", "lgbm", "hybrid", "xgboost"}
        else 0.0,
        description="Exponent applied to scale the sample weights. "
        "0=uniform weights, 1=linear scaling, >1=stronger emphasis on high values. "
        "Note: Defaults to 1.0 for gblinear congestion models.",
    )
    sample_weight_floor: float = Field(
        default=0.1,
        description="Minimum weight value to ensure all samples contribute to training.",
    )

    # Data splitting strategy
    data_splitter: DataSplitter = Field(
        default=DataSplitter(
            # Copied from OpenSTEF3 pipeline defaults
            val_fraction=0.15,
            test_fraction=0.0,
            stratification_fraction=0.15,
            min_days_for_stratification=4,
        ),
        description="Configuration for splitting data into training, validation, and test sets.",
    )

    # Evaluation
    evaluation_metrics: list[MetricProvider] = Field(
        default_factory=lambda: [R2Provider(), ObservedProbabilityProvider()],
        description="List of metric providers for evaluating model score.",
    )

    # Callbacks
    mlflow_storage: MLFlowStorage | None = Field(
        default_factory=MLFlowStorage,
        description="Configuration for MLflow experiment tracking and model storage.",
    )

    model_reuse_enable: bool = Field(
        default=True,
        description="Whether to enable reuse of previously trained models.",
    )
    model_reuse_max_age: timedelta = Field(
        default=timedelta(days=7),
        description="Maximum age of a model to be considered for reuse.",
    )

    model_selection_enable: bool = Field(
        default=True,
        description="Whether to enable automatic model selection based on performance.",
    )
    model_selection_metric: tuple[QuantileOrGlobal, str, MetricDirection] = Field(
        default=(Q(0.5), "R2", "higher_is_better"),
        description="Metric to monitor for model performance when retraining.",
    )
    model_selection_old_model_penalty: float = Field(
        default=1.2,
        description="Penalty to apply to the old model's metric to bias selection towards newer models.",
    )

    verbosity: Literal[0, 1, 2, 3, True] = Field(
        default=1, description="Verbosity level. 0=silent, 1=warning, 2=info, 3=debug"
    )

    # Metadata
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for the model.",
    )


def create_forecasting_workflow(
    config: ForecastingWorkflowConfig,
) -> CustomForecastingWorkflow:
    """Create a forecasting workflow from configuration.

    Builds a complete forecasting pipeline including preprocessing, forecaster, and postprocessing
    transforms based on the provided configuration.

    Args:
        config: Configuration object containing all workflow parameters.

    Returns:
        Configured forecasting workflow ready for training and prediction.

    Raises:
        ValueError: If an unsupported model type is specified.
    """
    checks = [
        InputConsistencyChecker(),
        FlatlineChecker(
            load_column=config.target_column,
            flatliner_threshold=config.flatliner_threshold,
            detect_non_zero_flatliner=config.detect_non_zero_flatliner,
            error_on_flatliner=False,
        ),
        CompletenessChecker(completeness_threshold=config.completeness_threshold),
    ]
    feature_adders = [
        LagsAdder(
            history_available=config.predict_history,
            horizons=config.horizons,
            add_trivial_lags=config.model not in {"gblinear", "hybrid"},  # GBLinear uses only 7day lag.
            target_column=config.target_column,
            custom_lags=[timedelta(days=7)] if config.model in {"gblinear", "hybrid"} else [],
        ),
        WindPowerFeatureAdder(
            windspeed_reference_column=config.wind_speed_column,
        ),
        AtmosphereDerivedFeaturesAdder(
            pressure_column=config.pressure_column,
            relative_humidity_column=config.relative_humidity_column,
            temperature_column=config.temperature_column,
        ),
        RadiationDerivedFeaturesAdder(
            coordinate=config.location.coordinate,
            radiation_column=config.radiation_column,
        ),
        CyclicFeaturesAdder(),
        DaylightFeatureAdder(
            coordinate=config.location.coordinate,
        ),
        RollingAggregatesAdder(
            feature=config.target_column,
            aggregation_functions=config.rolling_aggregate_features,
            horizons=config.horizons,
        ),
    ]
    feature_standardizers = [
        Clipper(
            selection=Include(config.energy_price_column).combine(config.clip_features),
            mode="standard",
        ),
        Scaler(selection=Exclude(config.target_column), method="standard"),
        SampleWeighter(
            target_column=config.target_column,
            weight_exponent=config.sample_weight_exponent,
            weight_floor=config.sample_weight_floor,
            weight_scale_percentile=config.sample_weight_scale_percentile,
        ),
        EmptyFeatureRemover(),
    ]

    if config.model == "xgboost":
        preprocessing = [
            *checks,
            *feature_adders,
            HolidayFeatureAdder(country_code=config.location.country_code),
            DatetimeFeaturesAdder(onehot_encode=False),
            *feature_standardizers,
        ]
        forecaster = XGBoostForecaster(
            config=XGBoostForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.xgboost_hyperparams,
                verbosity=config.verbosity,
            )
        )
        postprocessing = [QuantileSorter()]
    elif config.model == "lgbmlinear":
        preprocessing = [
            *checks,
            *feature_adders,
            HolidayFeatureAdder(country_code=config.location.country_code),
            DatetimeFeaturesAdder(onehot_encode=False),
            *feature_standardizers,
        ]
        forecaster = LGBMLinearForecaster(
            config=LGBMLinearForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.lgbmlinear_hyperparams,
            )
        )
        postprocessing = [QuantileSorter()]
    elif config.model == "lgbm":
        preprocessing = [
            *checks,
            *feature_adders,
            HolidayFeatureAdder(country_code=config.location.country_code),
            DatetimeFeaturesAdder(onehot_encode=False),
            *feature_standardizers,
        ]
        forecaster = LGBMForecaster(
            config=LGBMForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.lgbm_hyperparams,
            )
        )
        postprocessing = [QuantileSorter()]
    elif config.model == "gblinear":
        preprocessing = [
            *checks,
            *feature_adders,
            *feature_standardizers,
            Imputer(
                selection=Exclude(config.target_column),
                imputation_strategy="mean",
                fill_future_values=Include(config.energy_price_column),
            ),
            NaNDropper(
                selection=Exclude(config.target_column),
            ),
        ]
        forecaster = GBLinearForecaster(
            config=GBLinearForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.gblinear_hyperparams,
                verbosity=config.verbosity,
            ),
        )
        postprocessing = [QuantileSorter()]
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
    elif config.model == "hybrid":
        preprocessing = [
            *checks,
            *feature_adders,
            *feature_standardizers,
            Imputer(
                selection=Exclude(config.target_column),
                imputation_strategy="mean",
                fill_future_values=Include(config.energy_price_column),
            ),
            NaNDropper(
                selection=Exclude(config.target_column),
            ),
        ]
        forecaster = HybridForecaster(
            config=HybridForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.hybrid_hyperparams,
            )
        )
        postprocessing = [QuantileSorter()]
    else:
        msg = f"Unsupported model type: {config.model}"
        raise ValueError(msg)

    tags = {
        **config.location.tags,
        "model_type": config.model,
        **config.tags,
    }

    callbacks: list[ForecastingCallback] = []
    if config.mlflow_storage is not None:
        callbacks.append(
            MLFlowStorageCallback(
                storage=config.mlflow_storage,
                model_reuse_enable=config.model_reuse_enable,
                model_reuse_max_age=config.model_reuse_max_age,
                model_selection_enable=config.model_selection_enable,
                model_selection_metric=config.model_selection_metric,
                model_selection_old_model_penalty=config.model_selection_old_model_penalty,
            )
        )

    return CustomForecastingWorkflow(
        model=ForecastingModel(
            preprocessing=TransformPipeline(transforms=preprocessing),
            forecaster=forecaster,
            postprocessing=TransformPipeline(transforms=postprocessing),
            target_column=config.target_column,
            data_splitter=config.data_splitter,
            cutoff_history=config.cutoff_history,
            # Evaluation
            evaluation_metrics=config.evaluation_metrics,
            # Other
            tags=tags,
        ),
        model_id=config.model_id,
        callbacks=callbacks,
    )
