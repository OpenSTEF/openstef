# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level forecasting model that orchestrates the complete prediction pipeline.

Combines feature engineering, forecasting, and postprocessing into a unified interface.
Handles both single-horizon and multi-horizon forecasters while providing consistent
data transformation and validation.
"""

from datetime import datetime
from typing import Any, Self, cast, override

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from openstef_beam.evaluation import EvaluationConfig, EvaluationPipeline, SubsetMetric
from openstef_beam.evaluation.metric_providers import MetricProvider, ObservedProbabilityProvider, R2Provider
from openstef_core.base_model import BaseModel
from openstef_core.datasets import (
    ForecastDataset,
    ForecastInputDataset,
    MultiHorizon,
    TimeSeriesDataset,
    VersionedTimeSeriesDataset,
)
from openstef_core.datasets.data_split import DataSplitStrategy, StratifiedTrainTestSplitter
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.mixins import Predictor, State
from openstef_models.models.forecasting import Forecaster, HorizonForecaster
from openstef_models.transforms import FeatureEngineeringPipeline, PostprocessingPipeline


class ModelFitResult(BaseModel):
    input_dataset: VersionedTimeSeriesDataset | TimeSeriesDataset = Field()

    input_data_train: MultiHorizon[ForecastInputDataset] = Field()
    input_data_val: MultiHorizon[ForecastInputDataset] | None = Field(default=None)
    input_data_test: MultiHorizon[ForecastInputDataset] | None = Field(default=None)

    metrics_train: SubsetMetric = Field()
    metrics_val: SubsetMetric | None = Field(default=None)
    metrics_test: SubsetMetric | None = Field(default=None)
    metrics_full: SubsetMetric = Field()


class ForecastingModel(BaseModel, Predictor[VersionedTimeSeriesDataset | TimeSeriesDataset, ForecastDataset]):
    """Complete forecasting pipeline combining preprocessing, prediction, and postprocessing.

    Orchestrates the full forecasting workflow by managing feature engineering,
    model training/prediction, and result postprocessing. Automatically handles
    the differences between single-horizon and multi-horizon forecasters while
    ensuring data consistency and validation throughout the pipeline.

    Invariants:
        - fit() must be called before predict()
        - Forecaster and preprocessing horizons must match during initialization

    Example:
        Basic forecasting workflow:

        >>> from openstef_models.models.forecasting.constant_median_forecaster import (
        ...     ConstantMedianForecaster, ConstantMedianForecasterConfig
        ... )
        >>> from openstef_models.transforms import FeatureEngineeringPipeline
        >>> from openstef_core.types import LeadTime
        >>>
        >>> # Note: This is a conceptual example showing the API structure
        >>> # Real usage requires implemented forecaster classes
        >>> forecaster = ConstantMedianForecaster(
        ...     config=ConstantMedianForecasterConfig(horizons=[LeadTime.from_string("PT36H")])
        ... )
        >>> preprocessing = FeatureEngineeringPipeline(horizons=forecaster.config.horizons)
        >>>
        >>> # Create and train model
        >>> model = ForecastingModel(
        ...     forecaster=forecaster,
        ...     preprocessing=preprocessing
        ... )
        >>> model.fit(training_data)  # doctest: +SKIP
        >>>
        >>> # Generate forecasts
        >>> forecasts = model.predict(new_data)  # doctest: +SKIP
    """

    # Forecasting components
    preprocessing: FeatureEngineeringPipeline = Field(
        default=...,
        description="Feature engineering pipeline for transforming raw input data into model-ready features.",
    )
    forecaster: Forecaster | HorizonForecaster = Field(
        default=...,
        description="Underlying forecasting algorithm, either single-horizon or multi-horizon.",
    )
    postprocessing: PostprocessingPipeline[MultiHorizon[ForecastInputDataset], ForecastDataset] = Field(
        default_factory=PostprocessingPipeline[MultiHorizon[ForecastInputDataset], ForecastDataset],
        description="Postprocessing pipeline for transforming model outputs into final forecasts.",
    )
    target_column: str = Field(
        default="load",
        description="Name of the target variable column in datasets.",
    )
    # Evaluation - Data splitting configuration
    # Splits are applied in sequence: test_splitter first, then val_splitter on remaining data
    # Example: With defaults, 100% → 80% train+val / 20% test → 60% train / 20% val / 20% test
    split_strategy: DataSplitStrategy = Field(
        default_factory=lambda: DataSplitStrategy(
            test_splitter=StratifiedTrainTestSplitter(test_fraction=0.1),
            val_splitter=StratifiedTrainTestSplitter(test_fraction=0.15),
        ),
        description="Strategy for splitting data into train/validation/test sets. "
        "Handles explicit and automatic splitting scenarios.",
    )
    evaluation_metrics: list[MetricProvider] = Field(
        default_factory=lambda: [R2Provider(), ObservedProbabilityProvider()],
        description="List of metric providers for evaluating model score.",
    )
    # Metadata
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for the model.",
    )

    @model_validator(mode="after")
    def _validate_horizons_match(self) -> Self:
        if self.forecaster.config.horizons != self.preprocessing.horizons:
            message = (
                f"The forecaster horizons ({self.forecaster.config.horizons}) do not match the "
                f"preprocessing horizons ({self.preprocessing.horizons})."
            )
            raise ConfigurationError(message)

        return self

    @property
    @override
    def is_fitted(self) -> bool:
        return self.forecaster.is_fitted

    @override
    def fit(
        self,
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
        data_val: VersionedTimeSeriesDataset | TimeSeriesDataset | None = None,
        data_test: VersionedTimeSeriesDataset | TimeSeriesDataset | None = None,
    ) -> ModelFitResult:
        """Train the forecasting model on the provided dataset.

        Fits the preprocessing pipeline and underlying forecaster. Handles both
        single-horizon and multi-horizon forecasters appropriately.

        The data splitting follows this sequence:
        1. Split test set from full data (using test_splitter)
        2. Split validation from remaining train+val data (using val_splitter)
        3. Train on the final training set

        Args:
            data: Historical time series data with features and target values.
            data_val: Optional validation data. If provided, splitters are ignored for validation.
            data_test: Optional test data. If provided, splitters are ignored for test.

        Returns:
            FitResult containing training details and metrics.
        """
        # Fit the feature engineering transforms
        self.preprocessing.fit(data=data)

        # Transform the input data to a valid forecast input and split into train/val/test
        input_data_train, input_data_val, input_data_test = self._prepare_split_input(
            data=data, data_val=data_val, data_test=data_test
        )

        # Fit the model
        prediction = self._fit_forecaster(data=input_data_train, data_val=input_data_val)

        # Fit the postprocessing transforms
        self.postprocessing.fit(data=(input_data_train, prediction))

        # Calculate training, validation, test, and full metrics
        target_dataset = _dataset_to_target(data=data, target_column=self.target_column)

        def predict_and_score(input_data: MultiHorizon[ForecastInputDataset]) -> SubsetMetric:
            prediction = self._predict_forecaster(data=input_data)
            prediction = self.postprocessing.transform(data=(input_data, prediction))
            return self._calculate_score(ground_truth=target_dataset, prediction=prediction)

        metrics_train = predict_and_score(input_data_train)
        metrics_val = predict_and_score(input_data_val) if input_data_val else None
        metrics_test = predict_and_score(input_data_test) if input_data_test else None
        metrics_full = self.score(data=data)

        return ModelFitResult(
            input_dataset=data,
            input_data_train=input_data_train,
            input_data_val=input_data_val,
            input_data_test=input_data_test,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            metrics_test=metrics_test,
            metrics_full=metrics_full,
        )

    @override
    def predict(
        self, data: VersionedTimeSeriesDataset | TimeSeriesDataset, forecast_start: datetime | None = None
    ) -> ForecastDataset:
        """Generate forecasts using the trained model.

        Transforms input data through the preprocessing pipeline, generates predictions
        using the underlying forecaster, and applies postprocessing transformations.

        Args:
            data: Input time series data for generating forecasts.
            forecast_start: Starting time for forecasts. If None, uses data end time.

        Returns:
            Processed forecast dataset with predictions and uncertainty estimates.

        Raises:
            NotFittedError: If the model hasn't been trained yet.
        """
        if not self.is_fitted:
            raise NotFittedError(type(self.forecaster).__name__)

        # Transform the input data to a valid forecast input
        input_data = self._prepare_input(data=data, forecast_start=forecast_start)

        # Generate predictions
        raw_forecasts = self._predict_forecaster(data=input_data)

        return self.postprocessing.transform(data=(input_data, raw_forecasts))

    def _fit_forecaster(
        self,
        data: MultiHorizon[ForecastInputDataset],
        data_val: MultiHorizon[ForecastInputDataset] | None = None,
    ) -> ForecastDataset:
        if isinstance(self.forecaster, Forecaster):
            prediction = self.forecaster.fit_predict(data=data, data_val=data_val)
        else:
            horizon_input_data = data[self.preprocessing.horizons[0]]
            horizon_input_data_val = data_val[self.preprocessing.horizons[0]] if data_val else None
            prediction = self.forecaster.fit_predict(data=horizon_input_data, data_val=horizon_input_data_val)

        return prediction

    def _predict_forecaster(self, data: MultiHorizon[ForecastInputDataset]) -> ForecastDataset:
        if isinstance(self.forecaster, Forecaster):
            prediction = self.forecaster.predict(data=data)
        else:
            horizon_input_data = data[self.preprocessing.horizons[0]]
            prediction = self.forecaster.predict(data=horizon_input_data)

        return prediction

    def _prepare_split_input(
        self,
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
        data_val: VersionedTimeSeriesDataset | TimeSeriesDataset | None = None,
        data_test: VersionedTimeSeriesDataset | TimeSeriesDataset | None = None,
    ) -> tuple[
        MultiHorizon[ForecastInputDataset],
        MultiHorizon[ForecastInputDataset] | None,
        MultiHorizon[ForecastInputDataset] | None,
    ]:
        """Prepare and split input data into train, validation, and test sets.

        Args:
            data: Full dataset to split.
            data_val: Optional pre-split validation data.
            data_test: Optional pre-split test data.

        Returns:
            Tuple of (train_data, val_data, test_data) where val_data and test_data may be None.
        """
        # Preprocess all data
        preprocessed_data = self.preprocessing.transform(data=data)
        preprocessed_val = self.preprocessing.transform(data=data_val) if data_val else None
        preprocessed_test = self.preprocessing.transform(data=data_test) if data_test else None

        # Apply splitting strategy
        split_result = self.split_strategy.split_multihorizon(
            data=preprocessed_data,
            data_val=preprocessed_val,
            data_test=preprocessed_test,
        )

        # Convert to ForecastInputDataset with restored target column
        return (
            self._to_forecast_input(split_result.train, data),
            self._to_forecast_input(split_result.val, data) if split_result.val else None,
            self._to_forecast_input(split_result.test, data) if split_result.test else None,
        )

    def _to_forecast_input(
        self,
        data: MultiHorizon[TimeSeriesDataset],
        original_data: VersionedTimeSeriesDataset | TimeSeriesDataset,
    ) -> MultiHorizon[ForecastInputDataset]:
        """Convert preprocessed data to ForecastInputDataset with restored target column.

        Returns:
            Multi-horizon forecast input dataset with target column restored from original data.
        """
        target_dataset = _dataset_to_target(data=original_data, target_column=self.target_column)
        target_series = target_dataset.data[self.target_column]

        input_data = _dataset_to_input(data)
        return input_data.map_horizons(
            lambda dataset: ForecastInputDataset(
                data=dataset.data.assign(**{self.target_column: target_series}),
                sample_interval=dataset.sample_interval,
                target_column=self.target_column,
            )
        )

    def _prepare_input(
        self,
        data: VersionedTimeSeriesDataset | TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> MultiHorizon[ForecastInputDataset]:
        # Extract targets series to restore it later to ensure it is unchanged
        target_dataset = _dataset_to_target(data=data, target_column=self.target_column)
        target_series = target_dataset.data[self.target_column]

        # Make sure future target values are NaN
        if forecast_start is not None:
            mask = cast(pd.Series, target_series.index) >= forecast_start
            target_series = target_series.mask(cond=mask, other=np.nan)  # type: ignore

        # Apply preprocessing transforms
        input_data = self.preprocessing.transform(data=data)

        return input_data.map_horizons(
            lambda dataset: ForecastInputDataset(
                # Reassign target column to ensure it exists and is unchanged
                data=dataset.data.assign(**{self.target_column: target_series}),
                sample_interval=data.sample_interval,
                target_column=self.target_column,
                forecast_start=forecast_start,
            )
        )

    def score(
        self,
        data: TimeSeriesDataset | VersionedTimeSeriesDataset,
    ) -> SubsetMetric:
        """Evaluate model performance on the provided dataset.

        Generates predictions for the dataset and calculates evaluation metrics
        by comparing against ground truth values. Uses the configured evaluation
        metrics to assess forecast quality at the maximum forecast horizon.

        Args:
            data: Time series dataset containing both features and target values
                for evaluation.

        Returns:
            Evaluation metrics including configured providers (e.g., R2, observed
            probability) computed at the maximum forecast horizon.
        """
        ground_truth = _dataset_to_target(data=data, target_column=self.target_column)

        prediction = self.predict(data=data)

        return self._calculate_score(ground_truth=ground_truth, prediction=prediction)

    def _calculate_score(
        self,
        ground_truth: ForecastInputDataset,
        prediction: ForecastDataset,
    ) -> SubsetMetric:
        pipeline = EvaluationPipeline(
            # Needs only one horizon since we are using only a single prediction step
            # If a more comprehensive test is needed, a backtest should be run.
            config=EvaluationConfig(available_ats=[], lead_times=[self.forecaster.config.max_horizon]),
            quantiles=self.forecaster.config.quantiles,
            # Similarly windowed metrics are not relevant for single predictions.
            window_metric_providers=[],
            global_metric_providers=self.evaluation_metrics,
        )

        evaluation_result = pipeline.run_for_subset(
            filtering=self.forecaster.config.max_horizon,
            ground_truth=ground_truth,
            predictions=prediction,
        )
        global_metric = evaluation_result.get_global_metric()
        if not global_metric:
            return SubsetMetric(
                window="global",
                timestamp=ground_truth.index.min().to_pydatetime(),  # type: ignore
                metrics={},
            )

        return global_metric

    @override
    def to_state(self) -> State:
        return {
            "target_column": self.target_column,
            "preprocessing": self.preprocessing.to_state(),
            "forecaster": self.forecaster.to_state(),
            "postprocessing": self.postprocessing.to_state(),
        }

    @override
    def from_state(self, state: State) -> Self:
        state = cast(dict[str, Any], state)
        return self.__class__(
            target_column=state["target_column"],
            preprocessing=self.preprocessing.from_state(state["preprocessing"]),
            forecaster=self.forecaster.from_state(state["forecaster"]),
            postprocessing=self.postprocessing.from_state(state["postprocessing"]),
        )


def _dataset_to_input(data: MultiHorizon[TimeSeriesDataset]) -> MultiHorizon[ForecastInputDataset]:
    return data.map_horizons(
        lambda dataset: ForecastInputDataset(
            data=dataset.data,
            sample_interval=dataset.sample_interval,
            target_column=dataset.feature_names[0],  # Assume first column is target
        )
    )


def _dataset_to_target(
    data: TimeSeriesDataset | VersionedTimeSeriesDataset, target_column: str
) -> ForecastInputDataset:
    if isinstance(data, VersionedTimeSeriesDataset):
        return ForecastInputDataset(
            data=data.select_version().data[[target_column]],
            sample_interval=data.sample_interval,
            target_column=target_column,
            is_sorted=True,
        )
    return ForecastInputDataset(
        data=data.data[[target_column]],
        sample_interval=data.sample_interval,
        target_column=target_column,
        is_sorted=True,
    )


__all__ = ["ForecastingModel"]
