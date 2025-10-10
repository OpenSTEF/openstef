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
from openstef_core.datasets.data_split import BaseTrainTestSplitter, StratifiedTrainTestSplitter
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.mixins import Predictor, State, TransformPipeline
from openstef_models.models.forecasting import Forecaster, HorizonForecaster
from openstef_models.transforms import FeatureEngineeringPipeline


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
    postprocessing: TransformPipeline[ForecastDataset] = Field(
        default_factory=TransformPipeline[ForecastDataset],
        description="Postprocessing pipeline for transforming model outputs into final forecasts.",
    )
    target_column: str = Field(
        default="load",
        description="Name of the target variable column in datasets.",
    )
    # Evaluation
    train_test_splitter: BaseTrainTestSplitter | None = Field(default_factory=StratifiedTrainTestSplitter)
    train_val_splitter: BaseTrainTestSplitter | None = Field(default=None)
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
    ) -> ModelFitResult:
        """Train the forecasting model on the provided dataset.

        Fits the preprocessing pipeline and underlying forecaster. Handles both
        single-horizon and multi-horizon forecasters appropriately.

        Args:
            data: Historical time series data with features and target values.
            data_val: The validation data to evaluate and tune the predictor on (optional).

        Returns:
            FitResult containing training details and metrics.
        """
        # Fit the feature engineering transforms
        self.preprocessing.fit(data=data)

        # Transform the input data to a valid forecast input
        input_data_train, input_data_val = self._prepare_split_input(data=data, data_val=data_val)

        # Fit the model
        prediction = self._fit_forecaster(data=input_data_train, data_val=input_data_val)

        # Fit the postprocessing transforms
        self.postprocessing.fit(data=prediction)

        # Calculate training and operational metrics
        target_dataset = _dataset_to_target(data=data, target_column=self.target_column)

        predictions_train = self._predict_forecaster(data=input_data_train).pipe(self.postprocessing.transform)
        predictions_val = (
            self._predict_forecaster(data=input_data_val).pipe(self.postprocessing.transform)
            if input_data_val
            else None
        )

        metrics_train = self._calculate_score(
            ground_truth=target_dataset,
            prediction=predictions_train,
        )
        metrics_val = (
            self._calculate_score(
                ground_truth=target_dataset,
                prediction=predictions_val,
            )
            if predictions_val
            else None
        )
        metrics_full = self.score(data=data)

        return ModelFitResult(
            input_dataset=data,
            input_data_train=input_data_train,
            input_data_val=input_data_val,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
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

        return self.postprocessing.transform(raw_forecasts)

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
    ) -> tuple[MultiHorizon[ForecastInputDataset], MultiHorizon[ForecastInputDataset] | None]:
        # Transform the input data to a valid forecast input
        input_data_train = self._prepare_input(data=data)

        # Create or reuse validation data
        if data_val is not None:
            input_data_val = self._prepare_input(data=data_val)
        elif self.train_test_splitter is not None:
            if not self.train_test_splitter.is_fitted:
                self.train_test_splitter.fit_multihorizon(data=input_data_train)
            split_data_train, split_data_val = self.train_test_splitter.transform_multihorizon(input_data_train)
            input_data_train, input_data_val = (
                _dataset_to_input(split_data_train),
                _dataset_to_input(split_data_val),
            )
        else:
            input_data_val = None

        return input_data_train, input_data_val

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
