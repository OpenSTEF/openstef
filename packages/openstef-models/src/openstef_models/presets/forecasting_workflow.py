# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecasting workflow presets and configurations.

Provides predefined configurations and factory functions for common forecasting workflows,
including XGBoost, GBLinear, and Flatliner models with appropriate preprocessing pipelines.
"""

from datetime import timedelta
from decimal import Decimal
from typing import Any, Literal

import optuna
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
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q, Quantile, QuantileOrGlobal
from openstef_models.integrations.mlflow import MLFlowStorage, MLFlowStorageCallback
from openstef_models.mixins import ModelIdentifier
from openstef_models.models import ForecastingModel
from openstef_models.models.forecasting.flatliner_forecaster import FlatlinerForecaster
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearForecaster
from openstef_models.models.forecasting.median_forecaster import MedianForecaster
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.transforms.energy_domain import WindPowerFeatureAdder
from openstef_models.transforms.general import (
    Clipper,
    EmptyFeatureRemover,
    Imputer,
    NaNDropper,
    SampleWeighter,
    Scaler,
    Selector,
)
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
from openstef_models.utils.tuning import get_search_space, run_optuna_study, suggest_hyperparams
from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow, ForecastingCallback


class LocationConfig(BaseConfig):
    """Configuration for location information in forecasting workflows."""

    name: str = Field(default="test_location", description="Name of the forecasting location or workflow.")
    description: str = Field(default="", description="Description of the forecasting workflow.")
    coordinate: Coordinate = Field(
        default=Coordinate(
            latitude=Latitude(Decimal("52.132633")),
            longitude=Longitude(Decimal("5.291266")),
        ),
        description="Geographic coordinate of the location.",
    )
    country_code: CountryAlpha2 = Field(
        default=CountryAlpha2("NL"), description="Country code for holiday feature generation."
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
    run_name: str | None = Field(
        default=None, description="Optional name for this workflow run, can be used for versioning."
    )

    # Model configuration
    model: Literal["xgboost", "gblinear", "flatliner", "median"] = Field(
        description="Type of forecasting model to use."
    )
    quantiles: list[Quantile] = Field(
        default=[Q(0.5)], description="List of quantiles to predict for probabilistic forecasting."
    )

    sample_interval: timedelta = Field(
        default=timedelta(minutes=15), description="Time interval between consecutive data samples."
    )
    horizons: list[LeadTime] = Field(
        default=[LeadTime.from_string("PT48H")], description="List of forecast horizons to predict."
    )

    xgboost_hyperparams: XGBoostForecaster.HyperParams = Field(
        default=XGBoostForecaster.HyperParams(), description="Hyperparameters for XGBoost forecaster."
    )
    gblinear_hyperparams: GBLinearForecaster.HyperParams = Field(
        default=GBLinearForecaster.HyperParams(), description="Hyperparameters for GBLinear forecaster."
    )

    location: LocationConfig = Field(
        default=LocationConfig(), description="Location information for the forecasting workflow."
    )

    # Data properties
    target_column: str = Field(default="load", description="Name of the target variable column in datasets.")
    energy_price_column: str = Field(
        default="day_ahead_electricity_price", description="Name of the energy price column in datasets."
    )
    radiation_column: str = Field(default="radiation", description="Name of the radiation column in datasets.")
    wind_speed_column: str = Field(default="windspeed", description="Name of the wind speed column in datasets.")
    pressure_column: str = Field(default="pressure", description="Name of the pressure column in datasets.")
    temperature_column: str = Field(default="temperature", description="Name of the temperature column in datasets.")
    relative_humidity_column: str = Field(
        default="relative_humidity", description="Name of the relative humidity column in datasets."
    )
    selected_features: FeatureSelection = Field(
        default=FeatureSelection.ALL,
        description="Feature selection for which features to include/exclude.",
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
    predict_nonzero_flatliner: bool = Field(
        default=False,
        description="If True, predict the median of load measurements instead of zero (only for flatliner model).",
    )

    # Feature engineering
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
        default_factory=lambda data: 1.0 if data.get("model") == "gblinear" else 0.0,
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
        default_factory=MLFlowStorage, description="Configuration for MLflow experiment tracking and model storage."
    )

    model_reuse_enable: bool = Field(default=True, description="Whether to enable reuse of previously trained models.")
    model_reuse_max_age: timedelta = Field(
        default=timedelta(days=7), description="Maximum age of a model to be considered for reuse."
    )

    model_selection_enable: bool = Field(
        default=True, description="Whether to enable automatic model selection based on performance."
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

    # Hyperparameter tuning (Optuna)
    optuna_n_trials: int = Field(
        default=20,
        description="Number of Optuna trials to run when any search-space field has tune=True.",
    )
    optuna_seed: int | None = Field(
        default=42,
        description="Random seed for the Optuna TPE sampler.  Set to None to disable seeding.",
    )

    # Metadata
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for the model run.",
    )
    experiment_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for experiment tracking.",
    )


def create_forecasting_workflow(config: ForecastingWorkflowConfig) -> CustomForecastingWorkflow:
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
        Selector(selection=config.selected_features),
        InputConsistencyChecker(),
        FlatlineChecker(
            load_column=config.target_column,
            flatliner_threshold=config.flatliner_threshold,
            detect_non_zero_flatliner=config.detect_non_zero_flatliner,
            error_on_flatliner=True,
        ),
        CompletenessChecker(completeness_threshold=config.completeness_threshold),
    ]
    feature_adders = [
        LagsAdder(
            history_available=config.predict_history,
            horizons=config.horizons,
            add_trivial_lags=config.model != "gblinear",  # GBLinear uses only 7day lag.
            target_column=config.target_column,
            custom_lags=[timedelta(days=7)] if config.model == "gblinear" else [],
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
        Clipper(selection=Include(config.energy_price_column).combine(config.clip_features), mode="standard"),
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
        postprocessing = [
            QuantileSorter(),
            ConfidenceIntervalApplicator(
                quantiles=config.quantiles,
                add_quantiles_from_std=False,
            ),
        ]

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
        postprocessing = [
            QuantileSorter(),
            ConfidenceIntervalApplicator(
                quantiles=config.quantiles,
                add_quantiles_from_std=False,
            ),
        ]
    elif config.model == "median":
        preprocessing = [
            LagsAdder(
                history_available=config.predict_history,
                horizons=config.horizons,
                add_trivial_lags=True,
                target_column=config.target_column,
            )
        ]
        forecaster = MedianForecaster(
            config=MedianForecaster.Config(
                quantiles=config.quantiles,
                horizons=config.horizons,
            ),
        )
        postprocessing = []
    elif config.model == "flatliner":
        preprocessing = []
        forecaster = FlatlinerForecaster(
            config=FlatlinerForecaster.Config(
                quantiles=[Q(0.5)],
                horizons=config.horizons,
                predict_median=config.predict_nonzero_flatliner,
            )
        )
        postprocessing = [
            QuantileSorter(),
            ConfidenceIntervalApplicator(
                quantiles=[Q(0.5)],
                add_quantiles_from_std=False,
            ),
        ]
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
        run_name=config.run_name,
        callbacks=callbacks,
        experiment_tags=config.experiment_tags,
    )


class TuningResult:
    """Result of a :func:`fit_with_tuning` call.

    Attributes:
        workflow: The fitted :class:`CustomForecastingWorkflow`.
        fit_result: The :class:`ModelFitResult` from the final training run, or
            ``None`` if fitting was skipped (e.g. by an MLflow callback).
        study: The completed :class:`optuna.Study`.
        best_params: Flat dict of the best hyperparameter values found by Optuna,
            or an empty dict when tuning was not performed.
    """

    def __init__(
        self,
        workflow: CustomForecastingWorkflow,
        fit_result: ModelFitResult | None,
        study: optuna.Study | None,
        best_params: dict[str, Any],
    ) -> None:
        """Initialize a TuningResult with workflow and tuning outcomes.

        Args:
            workflow: The fitted forecasting workflow.
            fit_result: The result from the final training run, or None if fitting was skipped.
            study: The completed Optuna study, or None if no tuning was performed.
            best_params: Dictionary of best hyperparameter values found, or empty if no tuning.
        """
        self.workflow = workflow
        self.fit_result = fit_result
        self.study = study
        self.best_params = best_params

    def __repr__(self) -> str:
        """Return a string representation of the TuningResult."""
        tuned = f"{len(self.best_params)} params tuned" if self.best_params else "no tuning"
        return f"TuningResult({tuned})"


def tune(
    config: ForecastingWorkflowConfig,
    train_dataset: TimeSeriesDataset,
) -> tuple[ForecastingWorkflowConfig, optuna.Study, dict[str, Any]]:
    """Run hyperparameter tuning for a forecasting workflow configuration.

    Inspects the ``xgboost_hyperparams`` / ``gblinear_hyperparams`` instance of
    *config* to determine which hyperparameters are marked for tuning
    (by passing a ``TuningRange(tune=True)`` as the field value).

    The metric maximised during the study is determined by
    :attr:`ForecastingWorkflowConfig.model_selection_metric`.

    Args:
        config: Workflow configuration.  Pass ``TuningRange(tune=True)`` objects as
            field values on ``xgboost_hyperparams`` / ``gblinear_hyperparams`` to mark
            fields for tuning.  ``optuna_n_trials`` and ``optuna_seed`` control the study.
        train_dataset: Dataset used for all trial fit calls.

    Returns:
        A tuple of:

        - The config updated with the best hyperparameters found.
        - The completed :class:`optuna.Study`.
        - A flat dict of the best hyperparameter values.

    Raises:
        ValueError: If the model type does not support tuning (e.g. ``flatliner``),
            or if the model supports tuning but no field has ``tune=True``
            in the hyperparams instance.
    """
    if config.model not in {"xgboost", "gblinear"}:
        msg = (
            f"Model type '{config.model}' does not support hyperparameter tuning. "
            "Use 'xgboost' or 'gblinear' and pass TuningRange(tune=True) as field values."
        )
        raise ValueError(msg)

    if config.model == "xgboost":
        current_hp = config.xgboost_hyperparams
        hp_field = "xgboost_hyperparams"
    else:  # gblinear
        current_hp = config.gblinear_hyperparams
        hp_field = "gblinear_hyperparams"

    # Build the effective search space
    space = get_search_space(current_hp)

    if not space:
        msg = (
            f"No tunable hyperparameters found on `{hp_field}`. "
            "Pass TuningRange(tune=True) objects as field values when constructing it, "
            "e.g. `n_estimators=IntRange(100, 800, tune=True)`."
        )
        raise ValueError(msg)

    # Build the Optuna objective
    target_quantile, metric_name, _ = config.model_selection_metric

    def _objective(trial: optuna.Trial) -> float:
        tuned_hp = suggest_hyperparams(trial, space, current_hp)
        tuned_config = config.model_copy(update={hp_field: tuned_hp})
        trial_workflow = create_forecasting_workflow(tuned_config)
        trial_result = trial_workflow.fit(train_dataset)
        if trial_result is None:
            return float("-inf")
        metrics = trial_result.metrics_val if trial_result.metrics_val is not None else trial_result.metrics_train
        score = metrics.get_metric(quantile=target_quantile, metric_name=metric_name)
        return float(score) if score is not None else float("-inf")

    # Run the study
    study = run_optuna_study(
        objective=_objective,
        n_trials=config.optuna_n_trials,
        seed=config.optuna_seed,
        study_name=f"tuning_{config.model_id}",
    )

    best_hp = current_hp.model_copy(update=study.best_params)
    best_config = config.model_copy(update={hp_field: best_hp})
    return best_config, study, study.best_params


def fit_with_tuning(
    config: ForecastingWorkflowConfig,
    train_dataset: TimeSeriesDataset,
) -> TuningResult:
    """Create, optionally tune, and fit a forecasting workflow in one call.

    Inspects the ``xgboost_hyperparams`` / ``gblinear_hyperparams`` instance of
    *config* to determine whether any hyperparameter is marked for tuning
    (by passing a ``TuningRange(tune=True)`` as the field value).

    * **One or more tunable fields** runs an Optuna Bayesian search for
      :attr:`ForecastingWorkflowConfig.optuna_n_trials` trials via :func:`tune`,
      then trains the final model with the best hyperparameters found.

    The metric maximised during the study is determined by
    :attr:`ForecastingWorkflowConfig.model_selection_metric`.

    Args:
        config: Workflow configuration.  Pass ``TuningRange(tune=True)`` objects as
            field values on ``xgboost_hyperparams`` / ``gblinear_hyperparams`` to mark
            fields for tuning.  ``optuna_n_trials`` and ``optuna_seed`` control the study.
        train_dataset: Dataset used for **all** workflow fit calls (both tuning
            trials and the final fit).

    Returns:
        :class:`TuningResult` with the fitted workflow, fit result, optional study,
        and the best hyperparameter values.

    Example::

        from openstef_models.presets import ForecastingWorkflowConfig, fit_with_tuning
        from openstef_models.utils.tuning import FloatRange, IntRange
        from openstef_core.types import LeadTime, Q

        config = ForecastingWorkflowConfig(
            model_id="demo",
            model="xgboost",
            quantiles=[Q(0.5), Q(0.1), Q(0.9)],
            horizons=[LeadTime.from_string("PT36H")],
            xgboost_hyperparams=XGBoostForecaster.HyperParams(
                n_estimators=IntRange(100, 500, tune=True),
                learning_rate=FloatRange(None, None, log=True, tune=True),
            ),
            optuna_n_trials=20,
            mlflow_storage=None,
        )
        result = fit_with_tuning(config, train_dataset)  # doctest: +SKIP
        print(result.best_params)  # doctest: +SKIP
    """
    tuned_config, study, best_params = tune(config, train_dataset)
    workflow = create_forecasting_workflow(tuned_config)
    result = workflow.fit(train_dataset)
    return TuningResult(workflow=workflow, fit_result=result, study=study, best_params=best_params)
