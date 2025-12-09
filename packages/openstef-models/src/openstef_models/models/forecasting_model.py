# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level forecasting model that orchestrates the complete prediction pipeline.

Combines feature engineering, forecasting, and postprocessing into a unified interface.
Handles both single-horizon and multi-horizon forecasters while providing consistent
data transformation and validation.
"""

import logging
from datetime import datetime, timedelta
from functools import partial
from typing import cast, override

import pandas as pd
from pydantic import Field, PrivateAttr

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
from openstef_core.mixins import Predictor, TransformPipeline
from openstef_models.models.forecasting import Forecaster
from openstef_models.models.forecasting.forecaster import ForecasterConfig
from openstef_models.utils.data_split import DataSplitter


class ModelFitResult(BaseModel):
    """Result of fitting a forecasting model.

    Contains the original input dataset, split datasets used for training/validation/testing,
    and evaluation metrics computed on each subset.
    """

    input_dataset: TimeSeriesDataset = Field(description="Original time series dataset used for model fitting.")

    input_data_train: ForecastInputDataset = Field(description="Training dataset after preprocessing and splitting.")
    input_data_val: ForecastInputDataset | None = Field(
        default=None, description="Validation dataset after preprocessing and splitting, or None if not used."
    )
    input_data_test: ForecastInputDataset | None = Field(
        default=None, description="Test dataset after preprocessing and splitting, or None if not used."
    )

    metrics_train: SubsetMetric = Field(description="Evaluation metrics computed on the training dataset.")
    metrics_val: SubsetMetric | None = Field(
        default=None, description="Evaluation metrics computed on the validation dataset, or None if not used."
    )
    metrics_test: SubsetMetric | None = Field(
        default=None, description="Evaluation metrics computed on the test dataset, or None if not used."
    )
    metrics_full: SubsetMetric = Field(description="Evaluation metrics computed on the full original dataset.")


class ForecastingModel(BaseModel, Predictor[TimeSeriesDataset, ForecastDataset]):
    """Complete forecasting pipeline combining preprocessing, prediction, and postprocessing.

    Orchestrates the full forecasting workflow by managing feature engineering,
    model training/prediction, and result postprocessing. Automatically handles
    the differences between single-horizon and multi-horizon forecasters while
    ensuring data consistency and validation throughout the pipeline.

    Invariants:
        - fit() must be called before predict()
        - Forecaster and preprocessing horizons must match during initialization

    Important:
        The `cutoff_history` parameter is crucial when using lag-based features in
        preprocessing. For example, a lag-14 transformation creates NaN values for
        the first 14 days of data. Set `cutoff_history` to exclude these incomplete
        rows from training. You must configure this manually based on your preprocessing
        pipeline since lags cannot be automatically inferred from the transforms.

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
        ...     cutoff_history=timedelta(days=14),  # Match your maximum lag in preprocessing
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
    data_splitter: DataSplitter = Field(
        default_factory=DataSplitter,
        description="Data splitting strategy for train/validation/test sets.",
    )
    cutoff_history: timedelta = Field(
        default=timedelta(days=0),
        description="Amount of historical data to exclude from training and prediction due to incomplete features "
        "from lag-based preprocessing. When using lag transforms (e.g., lag-14), the first N days contain NaN values. "
        "Set this to match your maximum lag duration (e.g., timedelta(days=14)). "
        "Default of 0 assumes no invalid rows are created by preprocessing.",
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

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @property
    def config(self) -> ForecasterConfig:
        """Returns the configuration of the underlying forecaster."""
        return self.forecaster.config

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
        input_data_train = self.prepare_input(data=data)
        input_data_val = self.prepare_input(data=data_val) if data_val else None
        input_data_test = self.prepare_input(data=data_test) if data_test else None

        # Drop target column nan's from training data. One can not train on missing targets.
        target_dropna = partial(pd.DataFrame.dropna, subset=[self.target_column])  # pyright: ignore[reportUnknownMemberType]
        input_data_train = input_data_train.pipe_pandas(target_dropna)
        input_data_val = input_data_val.pipe_pandas(target_dropna) if input_data_val else None
        input_data_test = input_data_test.pipe_pandas(target_dropna) if input_data_test else None

        # Transform the input data to a valid forecast input and split into train/val/test
        input_data_train, input_data_val, input_data_test = self.data_splitter.split_dataset(
            data=input_data_train, data_val=input_data_val, data_test=input_data_test, target_column=self.target_column
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
        input_data = self.prepare_input(data=data, forecast_start=forecast_start)

        # Generate predictions
        raw_predictions = self._predict(input_data=input_data)

        return self.postprocessing.transform(data=raw_predictions)

    def prepare_input(
        self,
        data: TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> ForecastInputDataset:
        """Prepare input data for forecasting by applying preprocessing and filtering.

        Transforms raw time series data through the preprocessing pipeline, restores
        the target column, and filters out incomplete historical data to ensure
        training quality.

        Args:
            data: Raw time series dataset to prepare for forecasting.
            forecast_start: Optional start time for forecasts. If provided and earlier
                than the cutoff time, overrides the cutoff for data filtering.

        Returns:
            Processed forecast input dataset ready for model prediction.
        """
        # Transform and restore target column
        input_data = self.preprocessing.transform(data=data)
        input_data = restore_target(dataset=input_data, original_dataset=data, target_column=self.target_column)

        # Cut away input history to avoid training on incomplete data
        input_data_start = cast("pd.Series[pd.Timestamp]", input_data.index).min().to_pydatetime()
        input_data_cutoff = input_data_start + self.cutoff_history
        if forecast_start is not None and forecast_start < input_data_cutoff:
            input_data_cutoff = forecast_start
            self._logger.warning(
                "Forecast start %s is after input data start + cutoff history %s. Using forecast start as cutoff.",
                forecast_start,
                input_data_cutoff,
            )
        input_data = input_data.filter_by_range(start=input_data_cutoff)

        return ForecastInputDataset.from_timeseries(
            dataset=input_data,
            target_column=self.target_column,
            forecast_start=forecast_start,
        )

    def _predict(self, input_data: ForecastInputDataset) -> ForecastDataset:
        # Predict and restore target column
        prediction = self.forecaster.predict(data=input_data)
        return restore_target(dataset=prediction, original_dataset=input_data, target_column=self.target_column)

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

        # We need to make sure there are no NaNs in the target label for metric calculation
        prediction = prediction.pipe_pandas(pd.DataFrame.dropna, subset=[self.target_column])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

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


def restore_target[T: TimeSeriesDataset](
    dataset: T,
    original_dataset: TimeSeriesDataset,
    target_column: str,
) -> T:
    """Restore the target column from the original dataset to the given dataset.

    Maps target values from the original dataset to the dataset using index alignment.
    Ensures the target column is present in the dataset for downstream processing.

    Args:
        dataset: Dataset to modify by adding the target column.
        original_dataset: Source dataset containing the target values.
        target_column: Name of the target column to restore.

    Returns:
        Dataset with the target column restored from the original dataset.
    """
    target_series = original_dataset.select_features([target_column]).select_version().data[target_column]

    def _transform_restore_target(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{str(target_series.name): df.index.map(target_series)})  # pyright: ignore[reportUnknownMemberType]

    return dataset.pipe_pandas(_transform_restore_target)


__all__ = ["ForecastingModel", "ModelFitResult", "restore_target"]
