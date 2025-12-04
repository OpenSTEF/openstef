# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Ensemble forecasting workflow preset.

Mimics OpenSTEF-models forecasting workflow with ensemble capabilities.
"""

from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from openstef_meta.transforms.selector import Selector
from pydantic import Field

from openstef_beam.evaluation.metric_providers import (
    MetricDirection,
    MetricProvider,
    ObservedProbabilityProvider,
    R2Provider,
)
from openstef_core.base_model import BaseConfig
from openstef_core.datasets.timeseries_dataset import TimeSeriesDataset
from openstef_core.mixins.transform import Transform, TransformPipeline
from openstef_core.types import LeadTime, Q, Quantile, QuantileOrGlobal
from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel
from openstef_meta.models.forecast_combiners.learned_weights_combiner import WeightsCombiner
from openstef_meta.models.forecast_combiners.rules_combiner import RulesCombiner
from openstef_meta.models.forecast_combiners.stacking_combiner import StackingCombiner
from openstef_meta.models.forecasting.residual_forecaster import ResidualForecaster
from openstef_models.integrations.mlflow import MLFlowStorage
from openstef_models.mixins.model_serializer import ModelIdentifier
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearForecaster
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster
from openstef_models.models.forecasting.lgbmlinear_forecaster import LGBMLinearForecaster
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster
from openstef_models.presets.forecasting_workflow import LocationConfig
from openstef_models.transforms.energy_domain import WindPowerFeatureAdder
from openstef_models.transforms.general import Clipper, EmptyFeatureRemover, SampleWeighter, Scaler
from openstef_models.transforms.general.imputer import Imputer
from openstef_models.transforms.general.nan_dropper import NaNDropper
from openstef_models.transforms.postprocessing import QuantileSorter
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
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow, ForecastingCallback

if TYPE_CHECKING:
    from openstef_models.models.forecasting.forecaster import Forecaster


class EnsembleWorkflowConfig(BaseConfig):
    """Configuration for ensemble forecasting workflows."""

    model_id: ModelIdentifier

    # Ensemble configuration
    ensemble_type: Literal["learned_weights", "stacking", "rules"] = Field(default="learned_weights")
    base_models: Sequence[Literal["lgbm", "gblinear", "xgboost", "lgbm_linear"]] = Field(default=["lgbm", "gblinear"])
    combiner_model: Literal["lgbm", "rf", "xgboost", "logistic", "gblinear"] = Field(default="lgbm")

    # Forecast configuration
    quantiles: list[Quantile] = Field(
        default=[Q(0.5)],
        description="List of quantiles to predict for probabilistic forecasting.",
    )

    sample_interval: timedelta = Field(
        default=timedelta(minutes=15),
        description="Time interval between consecutive data samples.",
    )
    horizons: list[LeadTime] = Field(
        default=[LeadTime.from_string("PT36H")],
        description="List of forecast horizons to predict.",
    )

    location: LocationConfig = Field(
        default=LocationConfig(),
        description="Location information for the forecasting workflow.",
    )

    # Forecaster hyperparameters
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

    residual_hyperparams: ResidualForecaster.HyperParams = Field(
        default=ResidualForecaster.HyperParams(),
        description="Hyperparameters for Residual forecaster.",
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
    forecaster_sample_weight_exponent: dict[str, float] = Field(
        default={"gblinear": 1.0, "lgbm": 0, "xgboost": 0, "lgbm_linear": 0},
        description="Exponent applied to scale the sample weights. "
        "0=uniform weights, 1=linear scaling, >1=stronger emphasis on high values. "
        "Note: Defaults to 1.0 for gblinear congestion models.",
    )

    forecast_combiner_sample_weight_exponent: float = Field(
        default=0,
        description="Exponent applied to scale the sample weights for the forecast combiner model. "
        "0=uniform weights, 1=linear scaling, >1=stronger emphasis on high values.",
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
        default=0, description="Verbosity level. 0=silent, 1=warning, 2=info, 3=debug"
    )

    # Metadata
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for the model.",
    )


# Build preprocessing components
def checks(config: EnsembleWorkflowConfig) -> list[Transform[TimeSeriesDataset, TimeSeriesDataset]]:
    return [
        InputConsistencyChecker(),
        FlatlineChecker(
            load_column=config.target_column,
            flatliner_threshold=config.flatliner_threshold,
            detect_non_zero_flatliner=config.detect_non_zero_flatliner,
            error_on_flatliner=False,
        ),
        CompletenessChecker(completeness_threshold=config.completeness_threshold),
    ]


def feature_adders(config: EnsembleWorkflowConfig) -> list[Transform[TimeSeriesDataset, TimeSeriesDataset]]:
    return [
        LagsAdder(
            history_available=config.predict_history,
            horizons=config.horizons,
            add_trivial_lags=True,
            target_column=config.target_column,
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


def feature_standardizers(config: EnsembleWorkflowConfig) -> list[Transform[TimeSeriesDataset, TimeSeriesDataset]]:
    return [
        Clipper(selection=Include(config.energy_price_column).combine(config.clip_features), mode="standard"),
        Scaler(selection=Exclude(config.target_column), method="standard"),
        EmptyFeatureRemover(),
    ]


def create_ensemble_workflow(config: EnsembleWorkflowConfig) -> CustomForecastingWorkflow:  # noqa: C901, PLR0912, PLR0915
    """Create an ensemble forecasting workflow from configuration.

    Args:
        config (EnsembleWorkflowConfig): Configuration for the ensemble workflow.

    Returns:
        CustomForecastingWorkflow: Configured ensemble forecasting workflow.

    Raises:
        ValueError: If an unsupported base model or combiner type is specified.
    """
    # Common preprocessing
    common_preprocessing = TransformPipeline(
        transforms=[
            *checks(config),
            *feature_adders(config),
            HolidayFeatureAdder(country_code=config.location.country_code),
            DatetimeFeaturesAdder(onehot_encode=False),
            *feature_standardizers(config),
        ]
    )

    # Build forecasters and their processing pipelines
    forecaster_preprocessing: dict[str, list[Transform[TimeSeriesDataset, TimeSeriesDataset]]] = {}
    forecasters: dict[str, Forecaster] = {}
    for model_type in config.base_models:
        if model_type == "lgbm":
            forecasters[model_type] = LGBMForecaster(
                config=LGBMForecaster.Config(quantiles=config.quantiles, horizons=config.horizons)
            )
            forecaster_preprocessing[model_type] = [
                SampleWeighter(
                    target_column=config.target_column,
                    weight_exponent=config.forecaster_sample_weight_exponent[model_type],
                    weight_floor=config.sample_weight_floor,
                    weight_scale_percentile=config.sample_weight_scale_percentile,
                ),
            ]

        elif model_type == "gblinear":
            forecasters[model_type] = GBLinearForecaster(
                config=GBLinearForecaster.Config(quantiles=config.quantiles, horizons=config.horizons)
            )
            forecaster_preprocessing[model_type] = [
                SampleWeighter(
                    target_column=config.target_column,
                    weight_exponent=config.forecaster_sample_weight_exponent[model_type],
                    weight_floor=config.sample_weight_floor,
                    weight_scale_percentile=config.sample_weight_scale_percentile,
                ),
                Selector(
                    selection=FeatureSelection(
                        exclude={
                            "load_lag_P14D",
                            "load_lag_P13D",
                            "load_lag_P12D",
                            "load_lag_P11D",
                            "load_lag_P10D",
                            "load_lag_P9D",
                            "load_lag_P8D",
                            "load_lag_P7D",
                            "load_lag_P6D",
                            "load_lag_P5D",
                            "load_lag_P4D",
                            "load_lag_P3D",
                            "load_lag_P2D",
                        }
                    )
                ),
                Imputer(
                    selection=Exclude(config.target_column),
                    imputation_strategy="mean",
                    fill_future_values=Include(config.energy_price_column),
                ),
                NaNDropper(
                    selection=Exclude(config.target_column),
                ),
            ]
        elif model_type == "xgboost":
            forecasters[model_type] = XGBoostForecaster(
                config=XGBoostForecaster.Config(quantiles=config.quantiles, horizons=config.horizons)
            )
            forecaster_preprocessing[model_type] = [
                SampleWeighter(
                    target_column=config.target_column,
                    weight_exponent=config.forecaster_sample_weight_exponent[model_type],
                    weight_floor=config.sample_weight_floor,
                    weight_scale_percentile=config.sample_weight_scale_percentile,
                ),
            ]
        elif model_type == "lgbm_linear":
            forecasters[model_type] = LGBMLinearForecaster(
                config=LGBMLinearForecaster.Config(quantiles=config.quantiles, horizons=config.horizons)
            )
            forecaster_preprocessing[model_type] = [
                SampleWeighter(
                    target_column=config.target_column,
                    weight_exponent=config.forecaster_sample_weight_exponent[model_type],
                    weight_floor=config.sample_weight_floor,
                    weight_scale_percentile=config.sample_weight_scale_percentile,
                ),
            ]
        else:
            msg = f"Unsupported base model type: {model_type}"
            raise ValueError(msg)

    # Build combiner
    if config.ensemble_type == "learned_weights":
        if config.combiner_model == "lgbm":
            combiner_hp = WeightsCombiner.LGBMHyperParams()
        elif config.combiner_model == "rf":
            combiner_hp = WeightsCombiner.RFHyperParams()
        elif config.combiner_model == "xgboost":
            combiner_hp = WeightsCombiner.XGBHyperParams()
        elif config.combiner_model == "logistic":
            combiner_hp = WeightsCombiner.LogisticHyperParams()
        else:
            msg = f"Unsupported combiner model type: {config.combiner_model}"
            raise ValueError(msg)
        combiner_config = WeightsCombiner.Config(
            hyperparams=combiner_hp, horizons=config.horizons, quantiles=config.quantiles
        )
        combiner = WeightsCombiner(
            config=combiner_config,
        )
    elif config.ensemble_type == "stacking":
        if config.combiner_model == "lgbm":
            combiner_hp = StackingCombiner.LGBMHyperParams()
        elif config.combiner_model == "gblinear":
            combiner_hp = StackingCombiner.GBLinearHyperParams()
        else:
            msg = f"Unsupported combiner model type for stacking: {config.combiner_model}"
            raise ValueError(msg)
        combiner_config = StackingCombiner.Config(
            hyperparams=combiner_hp, horizons=config.horizons, quantiles=config.quantiles
        )
        combiner = StackingCombiner(
            config=combiner_config,
        )
    elif config.ensemble_type == "rules":
        combiner_config = RulesCombiner.Config(horizons=config.horizons, quantiles=config.quantiles)
        combiner = RulesCombiner(
            config=combiner_config,
        )
    else:
        msg = f"Unsupported ensemble type: {config.ensemble_type}"
        raise ValueError(msg)

    postprocessing = [QuantileSorter()]

    model_specific_preprocessing: dict[str, TransformPipeline[TimeSeriesDataset]] = {
        name: TransformPipeline(transforms=transforms) for name, transforms in forecaster_preprocessing.items()
    }

    if config.forecast_combiner_sample_weight_exponent != 0:
        combiner_transforms = [
            SampleWeighter(
                target_column=config.target_column,
                weight_exponent=config.forecast_combiner_sample_weight_exponent,
                weight_floor=config.sample_weight_floor,
                weight_scale_percentile=config.sample_weight_scale_percentile,
            ),
            Selector(selection=Include("sample_weight", config.target_column)),
        ]
    else:
        combiner_transforms = []

    combiner_preprocessing: TransformPipeline[TimeSeriesDataset] = TransformPipeline(transforms=combiner_transforms)

    ensemble_model = EnsembleForecastingModel(
        common_preprocessing=common_preprocessing,
        model_specific_preprocessing=model_specific_preprocessing,
        combiner_preprocessing=combiner_preprocessing,
        postprocessing=TransformPipeline(transforms=postprocessing),
        forecasters=forecasters,
        combiner=combiner,
        target_column=config.target_column,
        data_splitter=config.data_splitter,
    )

    callbacks: list[ForecastingCallback] = []
    # TODO(Egor): Implement MLFlow for OpenSTEF-meta # noqa: TD003

    return CustomForecastingWorkflow(model=ensemble_model, model_id=config.model_id, callbacks=callbacks)


__all__ = ["EnsembleWorkflowConfig", "create_ensemble_workflow"]
