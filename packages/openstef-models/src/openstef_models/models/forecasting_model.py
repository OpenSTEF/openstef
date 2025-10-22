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

import pandas as pd
from pydantic import Field

from openstef_beam.evaluation import EvaluationConfig, EvaluationPipeline, SubsetMetric
from openstef_beam.evaluation.metric_providers import MetricProvider, ObservedProbabilityProvider, R2Provider
from openstef_core.base_model import BaseModel
from openstef_core.datasets import (
    ForecastDataset,
    ForecastInputDataset,
    TimeSeriesDataset,
)
from openstef_core.datasets.timeseries_dataset import validate_horizons_present
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import Predictor, State, TransformPipeline
from openstef_models.models.forecasting import Forecaster
from openstef_models.utils.data_split import stratified_train_test_split, train_val_test_split


class ModelFitResult(BaseModel):
    input_dataset: TimeSeriesDataset = Field()

    input_data_train: ForecastInputDataset = Field()
    input_data_val: ForecastInputDataset | None = Field(default=None)
    input_data_test: ForecastInputDataset | None = Field(default=None)

    metrics_train: SubsetMetric = Field()
    metrics_val: SubsetMetric | None = Field(default=None)
    metrics_test: SubsetMetric | None = Field(default=None)
    metrics_full: SubsetMetric = Field()


class ForecastingModel(BaseModel, Predictor[TimeSeriesDataset, ForecastDataset]):
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
        >>> from openstef_core.types import LeadTime
        >>>
        >>> # Note: This is a conceptual example showing the API structure
        >>> # Real usage requires implemented forecaster classes
        >>> forecaster = ConstantMedianForecaster(
        ...     config=ConstantMedianForecasterConfig(horizons=[LeadTime.from_string("PT36H")])
        ... )
        >>> # Create and train model
        >>> model = ForecastingModel(
        ...     forecaster=forecaster,
        ... )
        >>> model.fit(training_data)  # doctest: +SKIP
        >>>
        >>> # Generate forecasts
        >>> forecasts = model.predict(new_data)  # doctest: +SKIP
    """

    # Forecasting components
    preprocessing: TransformPipeline[TimeSeriesDataset] = Field(
        default_factory=TransformPipeline[TimeSeriesDataset],
        description="Feature engineering pipeline for transforming raw input data into model-ready features.",
        exclude=True,
    )
    forecaster: Forecaster = Field(
        default=...,
        description="Underlying forecasting algorithm, either single-horizon or multi-horizon.",
        exclude=True,
    )
    postprocessing: TransformPipeline[ForecastDataset] = Field(
        default_factory=TransformPipeline[ForecastDataset],
        description="Postprocessing pipeline for transforming model outputs into final forecasts.",
        exclude=True,
    )
    target_column: str = Field(
        default="load",
        description="Name of the target variable column in datasets.",
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
    # Metadata
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for the model.",
    )

    @property
    @override
    def is_fitted(self) -> bool:
        return self.forecaster.is_fitted

    @override
    def fit(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
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
        validate_horizons_present(data, self.forecaster.config.horizons)

        # Fit the feature engineering transforms
        self.preprocessing.fit(data=data)

        # Transform and split input data
        input_data_train = self._prepare_input(data=data)
        input_data_val = self._prepare_input(data=data_val) if data_val else None
        input_data_test = self._prepare_input(data=data_test) if data_test else None

        # Transform the input data to a valid forecast input and split into train/val/test
        input_data_train, input_data_val, input_data_test = self._split_data(
            data=input_data_train, data_val=input_data_val, data_test=input_data_test
        )

        # Fit the model
        self.forecaster.fit(data=input_data_train, data_val=input_data_val)
        prediction_raw = self._predict(input_data=input_data_train)

        # Fit the postprocessing transforms
        prediction = self.postprocessing.fit_transform(data=prediction_raw)

        # Calculate training, validation, test, and full metrics
        def _predict_and_score(input_data: ForecastInputDataset) -> SubsetMetric:
            prediction_raw = self._predict(input_data=input_data)
            prediction = self.postprocessing.transform(data=prediction_raw)
            return self._calculate_score(prediction=prediction)

        metrics_train = self._calculate_score(prediction=prediction)
        metrics_val = _predict_and_score(input_data=input_data_val) if input_data_val else None
        metrics_test = _predict_and_score(input_data=input_data_test) if input_data_test else None
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
    def predict(self, data: TimeSeriesDataset, forecast_start: datetime | None = None) -> ForecastDataset:
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
        raw_predictions = self._predict(input_data=input_data)

        return self.postprocessing.transform(data=raw_predictions)

    def _split_data[T: TimeSeriesDataset](
        self,
        data: T,
        data_val: T | None = None,
        data_test: T | None = None,
    ) -> tuple[T, T | None, T | None]:
        """Prepare and split input data into train, validation, and test sets.

        Args:
            data: Full dataset to split.
            data_val: Optional pre-split validation data.
            data_test: Optional pre-split test data.

        Returns:
            Tuple of (train_data, val_data, test_data) where val_data and test_data may be None.
        """
        # Apply splitting strategy
        input_data_train, input_data_val, input_data_test = train_val_test_split(
            dataset=data,
            split_func=lambda dataset, fraction: stratified_train_test_split(
                dataset=dataset,
                test_fraction=fraction,
                stratification_fraction=self.stratification_fraction,
                target_column=self.target_column,
                random_state=42,
                min_days_for_stratification=self.min_days_for_stratification,
            ),
            val_fraction=self.val_fraction if data_val is None else 0.0,
            test_fraction=self.test_fraction if data_test is None else 0.0,
        )
        input_data_val = data_val or input_data_val
        input_data_test = data_test or input_data_test

        if input_data_val.index.empty:
            input_data_val = None
        if input_data_test.index.empty:
            input_data_test = None

        # Convert to ForecastInputDataset with restored target column
        return (input_data_train, input_data_val, input_data_test)

    def _prepare_input(
        self,
        data: TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> ForecastInputDataset:
        # Transform and restore target column
        input_data = self.preprocessing.transform(data=data)
        input_data = _restore_target(dataset=input_data, original_dataset=data, target_column=self.target_column)

        return ForecastInputDataset.from_timeseries(
            dataset=input_data,
            target_column=self.target_column,
            forecast_start=forecast_start,
        )

    def _predict(self, input_data: ForecastInputDataset) -> ForecastDataset:
        # Predict and restore target column
        prediction = self.forecaster.predict(data=input_data)
        return _restore_target(dataset=prediction, original_dataset=input_data, target_column=self.target_column)

    def score(
        self,
        data: TimeSeriesDataset,
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
        prediction = self.predict(data=data)

        return self._calculate_score(prediction=prediction)

    def _calculate_score(self, prediction: ForecastDataset) -> SubsetMetric:
        if prediction.target_series is None:
            raise ValueError("Prediction dataset must contain target series for scoring.")

        # Remove NaN values from target column before scoring
        prediction = prediction.pipe_pandas(lambda df: df.dropna(subset=[prediction.target_column]))  # pyright: ignore[reportUnknownMemberType]

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
            predictions=prediction,
        )
        global_metric = evaluation_result.get_global_metric()
        if not global_metric:
            return SubsetMetric(
                window="global",
                timestamp=prediction.forecast_start,
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


def _restore_target[T: TimeSeriesDataset](
    dataset: T,
    original_dataset: TimeSeriesDataset,
    target_column: str,
) -> T:
    target_series = original_dataset.select_features([target_column]).select_version().data[target_column]

    def _transform_restore_target(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{str(target_series.name): df.index.map(target_series)})  # pyright: ignore[reportUnknownMemberType]

    return dataset.pipe_pandas(_transform_restore_target)


__all__ = ["ForecastingModel"]
