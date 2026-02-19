# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Ensemble forecasting workflow preset.

Mimics OpenSTEF-models forecasting workflow with ensemble capabilities.
"""

from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, cast

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
from openstef_meta.models.forecast_combiners.learned_weights_combiner import (
    LGBMCombinerHyperParams,
    LogisticCombinerHyperParams,
    RFCombinerHyperParams,
    WeightsCombiner,
    XGBCombinerHyperParams,
)
from openstef_meta.models.forecast_combiners.stacking_combiner import (
    StackingCombiner,
)
from openstef_models.integrations.mlflow import MLFlowStorage, MLFlowStorageCallback
from openstef_models.mixins.model_serializer import ModelIdentifier
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearForecaster
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster
from openstef_models.models.forecasting.lgbmlinear_forecaster import LGBMLinearForecaster
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster
from openstef_models.presets.forecasting_workflow import LocationConfig
from openstef_models.transforms.energy_domain import WindPowerFeatureAdder
from openstef_models.transforms.general import Clipper, EmptyFeatureRemover, SampleWeightConfig, SampleWeighter, Scaler
from openstef_models.transforms.general.imputer import Imputer
from openstef_models.transforms.general.nan_dropper import NaNDropper
from openstef_models.transforms.general.selector import Selector
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
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)

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
        default=[LeadTime.from_string("PT48H")],
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

    # Learned weights combiner hyperparameters
    combiner_lgbm_hyperparams: LGBMCombinerHyperParams = Field(
        default=LGBMCombinerHyperParams(),
        description="Hyperparameters for LightGBM combiner.",
    )
    combiner_rf_hyperparams: RFCombinerHyperParams = Field(
        default=RFCombinerHyperParams(),
        description="Hyperparameters for Random Forest combiner.",
    )
    combiner_xgboost_hyperparams: XGBCombinerHyperParams = Field(
        default=XGBCombinerHyperParams(),
        description="Hyperparameters for XGBoost combiner.",
    )
    combiner_logistic_hyperparams: LogisticCombinerHyperParams = Field(
        default=LogisticCombinerHyperParams(),
        description="Hyperparameters for Logistic Regression combiner.",
    )

    # Stacking combiner hyperparameters
    combiner_stacking_lgbm_hyperparams: LGBMForecaster.HyperParams = Field(
        default=LGBMForecaster.HyperParams(),
        description="Hyperparameters for LightGBM stacking combiner.",
    )
    combiner_stacking_gblinear_hyperparams: GBLinearForecaster.HyperParams = Field(
        default=GBLinearForecaster.HyperParams(),
        description="Hyperparameters for GBLinear stacking combiner.",
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
    forecaster_sample_weights: dict[str, SampleWeightConfig] = Field(
        default={
            "gblinear": SampleWeightConfig(method="exponential", weight_exponent=1.0),
            "lgbm": SampleWeightConfig(weight_exponent=0.0),
            "xgboost": SampleWeightConfig(weight_exponent=0.0),
            "lgbm_linear": SampleWeightConfig(weight_exponent=0.0),
        },
        description="Per-forecaster sample weighting configuration. Use weight_exponent=0 to produce uniform weights.",
    )
    combiner_sample_weight: SampleWeightConfig = Field(
        default_factory=lambda: SampleWeightConfig(weight_exponent=0.0),
        description="Sample weighting configuration for the forecast combiner. "
        "Defaults to weight_exponent=0 (uniform weights).",
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
    return cast(
        list[Transform[TimeSeriesDataset, TimeSeriesDataset]],
        [
            Clipper(selection=Include(config.energy_price_column).combine(config.clip_features), mode="standard"),
            Scaler(selection=Exclude(config.target_column), method="standard"),
            EmptyFeatureRemover(),
        ],
    )


def create_ensemble_workflow(config: EnsembleWorkflowConfig) -> CustomForecastingWorkflow:  # noqa: C901, PLR0912
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
        sample_weight_config = config.forecaster_sample_weights.get(model_type, SampleWeightConfig())
        sample_weighter = SampleWeighter(config=sample_weight_config, target_column=config.target_column)

        if model_type == "lgbm":
            forecasters[model_type] = LGBMForecaster(
                config=LGBMForecaster.Config(
                    hyperparams=config.lgbm_hyperparams, quantiles=config.quantiles, horizons=config.horizons
                )
            )
            forecaster_preprocessing[model_type] = [sample_weighter]

        elif model_type == "gblinear":
            forecasters[model_type] = GBLinearForecaster(
                config=GBLinearForecaster.Config(
                    hyperparams=config.gblinear_hyperparams, quantiles=config.quantiles, horizons=config.horizons
                )
            )
            forecaster_preprocessing[model_type] = [
                sample_weighter,
                # Remove lags
                Selector(
                    selection=FeatureSelection(
                        exclude=set(
                            LagsAdder(
                                history_available=config.predict_history,
                                horizons=config.horizons,
                                add_trivial_lags=True,
                                target_column=config.target_column,
                            ).features_added()
                        ).difference({"load_lag_P7D"})
                    )
                ),
                # Remove holiday features to avoid linear dependencies
                Selector(
                    selection=FeatureSelection(
                        exclude=set(HolidayFeatureAdder(country_code=config.location.country_code).features_added())
                    )
                ),
                Selector(
                    selection=FeatureSelection(exclude=set(DatetimeFeaturesAdder(onehot_encode=False).features_added()))
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
                config=XGBoostForecaster.Config(
                    hyperparams=config.xgboost_hyperparams, quantiles=config.quantiles, horizons=config.horizons
                )
            )
            forecaster_preprocessing[model_type] = [sample_weighter]
        elif model_type == "lgbm_linear":
            forecasters[model_type] = LGBMLinearForecaster(
                config=LGBMLinearForecaster.Config(
                    hyperparams=config.lgbmlinear_hyperparams, quantiles=config.quantiles, horizons=config.horizons
                )
            )
            forecaster_preprocessing[model_type] = [sample_weighter]
        else:
            msg = f"Unsupported base model type: {model_type}"
            raise ValueError(msg)

    # Build combiner
    match (config.ensemble_type, config.combiner_model):
        case ("learned_weights", "lgbm"):
            combiner = WeightsCombiner(
                config=WeightsCombiner.Config(
                    hyperparams=config.combiner_lgbm_hyperparams, horizons=config.horizons, quantiles=config.quantiles
                )
            )
        case ("learned_weights", "rf"):
            combiner = WeightsCombiner(
                config=WeightsCombiner.Config(
                    hyperparams=config.combiner_rf_hyperparams, horizons=config.horizons, quantiles=config.quantiles
                )
            )
        case ("learned_weights", "xgboost"):
            combiner = WeightsCombiner(
                config=WeightsCombiner.Config(
                    hyperparams=config.combiner_xgboost_hyperparams,
                    horizons=config.horizons,
                    quantiles=config.quantiles,
                )
            )
        case ("learned_weights", "logistic"):
            combiner = WeightsCombiner(
                config=WeightsCombiner.Config(
                    hyperparams=config.combiner_logistic_hyperparams,
                    horizons=config.horizons,
                    quantiles=config.quantiles,
                )
            )
        case ("stacking", "lgbm"):
            combiner = StackingCombiner(
                config=StackingCombiner.Config(
                    hyperparams=config.combiner_stacking_lgbm_hyperparams,
                    horizons=config.horizons,
                    quantiles=config.quantiles,
                )
            )
        case ("stacking", "gblinear"):
            combiner = StackingCombiner(
                config=StackingCombiner.Config(
                    hyperparams=config.combiner_stacking_gblinear_hyperparams,
                    horizons=config.horizons,
                    quantiles=config.quantiles,
                )
            )
        case _:
            msg = f"Unsupported ensemble and combiner combination: {config.ensemble_type}, {config.combiner_model}"
            raise ValueError(msg)

    postprocessing = [QuantileSorter()]

    model_specific_preprocessing: dict[str, TransformPipeline[TimeSeriesDataset]] = {
        name: TransformPipeline(transforms=transforms) for name, transforms in forecaster_preprocessing.items()
    }

    combiner_transforms = [
        SampleWeighter(config=config.combiner_sample_weight, target_column=config.target_column),
        Selector(selection=Include("sample_weight", config.target_column)),
    ]

    combiner_preprocessing: TransformPipeline[TimeSeriesDataset] = TransformPipeline(transforms=combiner_transforms)

    tags = {
        **config.location.tags,
        "ensemble_type": config.ensemble_type,
        "combiner_model": config.combiner_model,
        "base_models": ",".join(config.base_models),
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
        model=EnsembleForecastingModel(
            common_preprocessing=common_preprocessing,
            model_specific_preprocessing=model_specific_preprocessing,
            combiner_preprocessing=combiner_preprocessing,
            postprocessing=TransformPipeline(transforms=postprocessing),
            forecasters=forecasters,
            combiner=combiner,
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


__all__ = ["EnsembleWorkflowConfig", "create_ensemble_workflow"]
