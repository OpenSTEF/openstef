# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
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
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner
from openstef_meta.utils.datasets import EnsembleForecastDataset
from openstef_models.models.forecasting import Forecaster
from openstef_models.models.forecasting.forecaster import ForecasterConfig
from openstef_models.models.forecasting_model import ModelFitResult
from openstef_models.utils.data_split import DataSplitter

logger = logging.getLogger(__name__)


class EnsembleModelFitResult(BaseModel):
    forecaster_fit_results: dict[str, ModelFitResult] = Field(description="ModelFitResult for each base Forecaster")

    combiner_fit_result: ModelFitResult = Field(description="ModelFitResult for the ForecastCombiner")

    # Make compatible with ModelFitResult interface
    @property
    def input_dataset(self) -> EnsembleForecastDataset:
        """Returns the input dataset used for fitting the combiner."""
        return cast(
            "EnsembleForecastDataset",
            self.combiner_fit_result.input_dataset,
        )

    @property
    def input_data_train(self) -> ForecastInputDataset:
        """Returns the training input data used for fitting the combiner."""
        return self.combiner_fit_result.input_data_train

    @property
    def input_data_val(self) -> ForecastInputDataset | None:
        """Returns the validation input data used for fitting the combiner."""
        return self.combiner_fit_result.input_data_val

    @property
    def input_data_test(self) -> ForecastInputDataset | None:
        """Returns the test input data used for fitting the combiner."""
        return self.combiner_fit_result.input_data_test

    @property
    def metrics_train(self) -> SubsetMetric:
        """Returns the full metrics calculated during combiner fitting."""
        return self.combiner_fit_result.metrics_train

    @property
    def metrics_val(self) -> SubsetMetric | None:
        """Returns the full metrics calculated during combiner fitting."""
        return self.combiner_fit_result.metrics_val

    @property
    def metrics_test(self) -> SubsetMetric | None:
        """Returns the full metrics calculated during combiner fitting."""
        return self.combiner_fit_result.metrics_test

    @property
    def metrics_full(self) -> SubsetMetric:
        """Returns the full metrics calculated during combiner fitting."""
        return self.combiner_fit_result.metrics_full


class EnsembleForecastingModel(BaseModel, Predictor[TimeSeriesDataset, ForecastDataset]):
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
        >>> from openstef_meta.models.forecast_combiners.learned_weights_combiner import WeightsCombiner
        >>> from openstef_core.types import LeadTime
        >>>
        >>> # Note: This is a conceptual example showing the API structure
        >>> # Real usage requires implemented forecaster classes
        >>> forecaster_1 = ConstantMedianForecaster(
        ...     config=ConstantMedianForecasterConfig(horizons=[LeadTime.from_string("PT36H")])
        ... )
        >>> forecaster_2 = ConstantMedianForecaster(
        ...     config=ConstantMedianForecasterConfig(horizons=[LeadTime.from_string("PT36H")])
        ... )
        >>> combiner_config = WeightsCombiner.Config(
        ...     horizons=[LeadTime.from_string("PT36H")],
        ... )
        >>> # Create and train model
        >>> model = EnsembleForecastingModel(
        ...     forecasters={"constant_median": forecaster_1, "constant_median_2": forecaster_2},
        ...     combiner=WeightsCombiner(config=combiner_config),
        ...     cutoff_history=timedelta(days=14),  # Match your maximum lag in preprocessing
        ... )
        >>> model.fit(training_data)  # doctest: +SKIP
        >>>
        >>> # Generate forecasts
        >>> forecasts = model.predict(new_data)  # doctest: +SKIP
    """

    # Forecasting components
    common_preprocessing: TransformPipeline[TimeSeriesDataset] = Field(
        default_factory=TransformPipeline[TimeSeriesDataset],
        description="Feature engineering pipeline for transforming raw input data into model-ready features.",
        exclude=True,
    )

    model_specific_preprocessing: dict[str, TransformPipeline[TimeSeriesDataset]] = Field(
        default_factory=dict,
        description="Feature engineering pipeline for transforming raw input data into model-ready features.",
        exclude=True,
    )

    forecasters: dict[str, Forecaster] = Field(
        default=...,
        description="Underlying forecasting algorithm, either single-horizon or multi-horizon.",
        exclude=True,
    )

    combiner: ForecastCombiner = Field(
        default=...,
        description="Combiner to aggregate forecasts from multiple forecasters if applicable.",
        exclude=True,
    )

    combiner_preprocessing: TransformPipeline[TimeSeriesDataset] = Field(
        default_factory=TransformPipeline[TimeSeriesDataset],
        description="Feature engineering for the forecast combiner.",
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
    def config(self) -> list[ForecasterConfig]:
        """Returns the configuration of the underlying forecaster."""
        return [x.config for x in self.forecasters.values()]

    @property
    @override
    def is_fitted(self) -> bool:
        return all(f.is_fitted for f in self.forecasters.values()) and self.combiner.is_fitted

    @property
    def forecaster_names(self) -> list[str]:
        """Returns the names of the underlying forecasters."""
        return list(self.forecasters.keys())

    @override
    def fit(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
    ) -> EnsembleModelFitResult:
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
        # Fit forecasters
        train_ensemble, val_ensemble, test_ensemble, forecaster_fit_results = self._fit_forecasters(
            data=data,
            data_val=data_val,
            data_test=data_test,
        )

        combiner_fit_result = self._fit_combiner(
            train_ensemble_dataset=train_ensemble,
            val_ensemble_dataset=val_ensemble,
            test_ensemble_dataset=test_ensemble,
            data=data,
            data_val=data_val,
            data_test=data_test,
        )

        return EnsembleModelFitResult(
            forecaster_fit_results=forecaster_fit_results,
            combiner_fit_result=combiner_fit_result,
        )

    @staticmethod
    def _combine_datasets(
        data: ForecastInputDataset, additional_features: ForecastInputDataset
    ) -> ForecastInputDataset:
        """Combine base learner predictions with additional features for final learner input.

        Args:
            data: ForecastInputDataset containing base learner predictions.
            additional_features: ForecastInputDataset containing additional features.

        Returns:
            ForecastInputDataset with combined features.
        """
        additional_df = additional_features.data.loc[
            :, [col for col in additional_features.data.columns if col not in data.data.columns]
        ]
        # Merge on index to combine datasets
        combined_df = data.data.join(additional_df)

        return ForecastInputDataset(
            data=combined_df,
            sample_interval=data.sample_interval,
            forecast_start=data.forecast_start,
        )

    def _transform_combiner_data(self, data: TimeSeriesDataset) -> ForecastInputDataset | None:
        if len(self.combiner_preprocessing.transforms) == 0:
            return None
        combiner_data = self.combiner_preprocessing.transform(data)
        return ForecastInputDataset.from_timeseries(combiner_data, target_column=self.target_column)

    def _fit_prepare_combiner_data(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
    ) -> tuple[ForecastInputDataset | None, ForecastInputDataset | None, ForecastInputDataset | None]:

        if len(self.combiner_preprocessing.transforms) == 0:
            return None, None, None
        self.combiner_preprocessing.fit(data=data)

        input_data_train = self.combiner_preprocessing.transform(data)
        input_data_val = self.combiner_preprocessing.transform(data_val) if data_val else None
        input_data_test = self.combiner_preprocessing.transform(data_test) if data_test else None

        input_data_train, input_data_val, input_data_test = self.data_splitter.split_dataset(
            data=input_data_train, data_val=input_data_val, data_test=input_data_test, target_column=self.target_column
        )
        combiner_data = ForecastInputDataset.from_timeseries(input_data_train, target_column=self.target_column)

        combiner_data_val = (
            ForecastInputDataset.from_timeseries(input_data_val, target_column=self.target_column)
            if input_data_val
            else None
        )

        combiner_data_test = (
            ForecastInputDataset.from_timeseries(input_data_test, target_column=self.target_column)
            if input_data_test
            else None
        )

        return combiner_data, combiner_data_val, combiner_data_test

    def _fit_forecasters(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
    ) -> tuple[
        EnsembleForecastDataset,
        EnsembleForecastDataset | None,
        EnsembleForecastDataset | None,
        dict[str, ModelFitResult],
    ]:

        predictions_train: dict[str, ForecastDataset] = {}
        predictions_val: dict[str, ForecastDataset | None] = {}
        predictions_test: dict[str, ForecastDataset | None] = {}
        results: dict[str, ModelFitResult] = {}

        if data_test is not None:
            logger.info("Data test provided during fit, but will be ignored for MetaForecating")

        # Fit the feature engineering transforms
        self.common_preprocessing.fit(data=data)
        data_transformed = self.common_preprocessing.transform(data=data)
        [
            self.model_specific_preprocessing[name].fit(data=data_transformed)
            for name in self.model_specific_preprocessing
        ]
        logger.debug("Completed fitting preprocessing pipelines.")

        # Fit the forecasters
        for name in self.forecasters:
            logger.debug("Started fitting Forecaster '%s'.", name)
            predictions_train[name], predictions_val[name], predictions_test[name], results[name] = (
                self._fit_forecaster(
                    data=data,
                    data_val=data_val,
                    data_test=None,
                    forecaster_name=name,
                )
            )

        train_ensemble = EnsembleForecastDataset.from_forecast_datasets(
            predictions_train, target_series=data.data[self.target_column]
        )

        if all(isinstance(v, ForecastDataset) for v in predictions_val.values()):
            val_ensemble = EnsembleForecastDataset.from_forecast_datasets(
                {k: v for k, v in predictions_val.items() if v is not None},
                target_series=data.data[self.target_column],
            )
        else:
            val_ensemble = None

        if all(isinstance(v, ForecastDataset) for v in predictions_test.values()):
            test_ensemble = EnsembleForecastDataset.from_forecast_datasets(
                {k: v for k, v in predictions_test.items() if v is not None},
                target_series=data.data[self.target_column],
            )
        else:
            test_ensemble = None

        return train_ensemble, val_ensemble, test_ensemble, results

    def _fit_forecaster(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
        forecaster_name: str = "",
    ) -> tuple[
        ForecastDataset,
        ForecastDataset | None,
        ForecastDataset | None,
        ModelFitResult,
    ]:
        """Train the forecaster on the provided dataset.

        Args:
            data: Historical time series data with features and target values.
            data_val: Optional validation data.
            data_test: Optional test data.
            forecaster_name: Name of the forecaster to train.

        Returns:
            ForecastDataset containing the trained forecaster's predictions.
        """
        forecaster = self.forecasters[forecaster_name]
        validate_horizons_present(data, forecaster.config.horizons)

        # Transform and split input data
        input_data_train = self.prepare_input(data=data, forecaster_name=forecaster_name)
        input_data_val = self.prepare_input(data=data_val, forecaster_name=forecaster_name) if data_val else None
        input_data_test = self.prepare_input(data=data_test, forecaster_name=forecaster_name) if data_test else None

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
        logger.debug("Started fitting forecaster '%s'.", forecaster_name)
        forecaster.fit(data=input_data_train, data_val=input_data_val)
        logger.debug("Completed fitting forecaster '%s'.", forecaster_name)

        prediction_train = self._predict_forecaster(input_data=input_data_train, forecaster_name=forecaster_name)
        metrics_train = self._calculate_score(prediction=prediction_train)

        if input_data_val is not None:
            prediction_val = self._predict_forecaster(input_data=input_data_val, forecaster_name=forecaster_name)
            metrics_val = self._calculate_score(prediction=prediction_val)
        else:
            prediction_val = None
            metrics_val = None

        if input_data_test is not None:
            prediction_test = self._predict_forecaster(input_data=input_data_test, forecaster_name=forecaster_name)
            metrics_test = self._calculate_score(prediction=prediction_test)
        else:
            prediction_test = None
            metrics_test = None

        result = ModelFitResult(
            input_dataset=input_data_train,
            input_data_train=input_data_train,
            input_data_val=input_data_val,
            input_data_test=input_data_test,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            metrics_test=metrics_test,
            metrics_full=metrics_train,
        )

        return prediction_train, prediction_val, prediction_test, result

    def _predict_forecaster(self, input_data: ForecastInputDataset, forecaster_name: str) -> ForecastDataset:
        # Predict and restore target column
        prediction_raw = self.forecasters[forecaster_name].predict(data=input_data)
        prediction = self.postprocessing.transform(prediction_raw)
        return restore_target(dataset=prediction, original_dataset=input_data, target_column=self.target_column)

    def _predict_forecasters(
        self,
        data: TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> EnsembleForecastDataset:
        predictions: dict[str, ForecastDataset] = {}
        for name in self.forecasters:
            logger.debug("Generating predictions for forecaster '%s'.", name)
            input_data = self.prepare_input(data=data, forecast_start=forecast_start, forecaster_name=name)
            predictions[name] = self._predict_forecaster(
                input_data=input_data,
                forecaster_name=name,
            )

        return EnsembleForecastDataset.from_forecast_datasets(predictions, target_series=data.data[self.target_column])

    def prepare_input(
        self,
        data: TimeSeriesDataset,
        forecaster_name: str = "",
        forecast_start: datetime | None = None,
    ) -> ForecastInputDataset:
        """Prepare input data for forecastingfiltering.

        Args:
            data: Raw time series dataset to prepare for forecasting.
            forecast_start: Optional start time for forecasts. If provided and earlier
                than the cutoff time, overrides the cutoff for data filtering.
            forecaster_name: Name of the forecaster for which to prepare input data.

        Returns:
            Processed forecast input dataset ready for model prediction.
        """
        logger.debug("Preparing input data for forecaster '%s'.", forecaster_name)
        input_data = restore_target(dataset=data, original_dataset=data, target_column=self.target_column)

        # Transform the data
        input_data = self.common_preprocessing.transform(data=input_data)
        if forecaster_name in self.model_specific_preprocessing:
            logger.debug("Applying model-specific preprocessing for forecaster '%s'.", forecaster_name)
            input_data = self.model_specific_preprocessing[forecaster_name].transform(data=input_data)

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

    def _predict_combiner(
        self, ensemble_dataset: EnsembleForecastDataset, original_data: TimeSeriesDataset
    ) -> ForecastDataset:
        logger.debug("Predicting combiner.")
        features = self._transform_combiner_data(data=original_data)
        prediction_raw = self.combiner.predict(ensemble_dataset, additional_features=features)
        prediction = self.postprocessing.transform(prediction_raw)

        prediction.data[ensemble_dataset.target_column] = ensemble_dataset.target_series
        return prediction

    def _fit_combiner(
        self,
        data: TimeSeriesDataset,
        train_ensemble_dataset: EnsembleForecastDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
        val_ensemble_dataset: EnsembleForecastDataset | None = None,
        test_ensemble_dataset: EnsembleForecastDataset | None = None,
    ) -> ModelFitResult:

        features_train, features_val, features_test = self._fit_prepare_combiner_data(
            data=data, data_val=data_val, data_test=data_test
        )

        logger.debug("Fitting combiner.")
        self.combiner.fit(
            data=train_ensemble_dataset, data_val=val_ensemble_dataset, additional_features=features_train
        )

        prediction_train = self.combiner.predict(train_ensemble_dataset, additional_features=features_train)
        metrics_train = self._calculate_score(prediction=prediction_train)

        if val_ensemble_dataset is not None:
            prediction_val = self.combiner.predict(val_ensemble_dataset, additional_features=features_val)
            metrics_val = self._calculate_score(prediction=prediction_val)
        else:
            prediction_val = None
            metrics_val = None

        if test_ensemble_dataset is not None:
            prediction_test = self.combiner.predict(test_ensemble_dataset, additional_features=features_test)
            metrics_test = self._calculate_score(prediction=prediction_test)
        else:
            prediction_test = None
            metrics_test = None

        return ModelFitResult(
            input_dataset=train_ensemble_dataset,
            input_data_train=train_ensemble_dataset.select_quantile(quantile=self.config[0].quantiles[0]),
            input_data_val=val_ensemble_dataset.select_quantile(quantile=self.config[0].quantiles[0])
            if val_ensemble_dataset
            else None,
            input_data_test=test_ensemble_dataset.select_quantile(quantile=self.config[0].quantiles[0])
            if test_ensemble_dataset
            else None,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            metrics_test=metrics_test,
            metrics_full=metrics_train,
        )

    def _predict_contributions_combiner(
        self, ensemble_dataset: EnsembleForecastDataset, original_data: TimeSeriesDataset
    ) -> pd.DataFrame:

        features = self._transform_combiner_data(data=original_data)
        return self.combiner.predict_contributions(ensemble_dataset, additional_features=features)

    def predict(self, data: TimeSeriesDataset, forecast_start: datetime | None = None) -> ForecastDataset:
        """Generate forecasts for the provided dataset.

        Args:
            data: Input time series dataset for prediction.
            forecast_start: Optional start time for forecasts.

        Returns:
            ForecastDataset containing the generated forecasts.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)
        logger.debug("Generating predictions.")

        ensemble_predictions = self._predict_forecasters(data=data, forecast_start=forecast_start)

        # Predict and restore target column
        prediction = self._predict_combiner(
            ensemble_dataset=ensemble_predictions,
            original_data=data,
        )
        return restore_target(dataset=prediction, original_dataset=data, target_column=self.target_column)

    def predict_contributions(self, data: TimeSeriesDataset, forecast_start: datetime | None = None) -> pd.DataFrame:
        """Generate forecasts for the provided dataset.

        Args:
            data: Input time series dataset for prediction.
            forecast_start: Optional start time for forecasts.

        Returns:
            ForecastDataset containing the generated forecasts.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        ensemble_predictions = self._predict_forecasters(data=data, forecast_start=forecast_start)

        return self._predict_contributions_combiner(
            ensemble_dataset=ensemble_predictions,
            original_data=data,
        )

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
            config=EvaluationConfig(available_ats=[], lead_times=[self.config[0].max_horizon]),
            quantiles=self.config[0].quantiles,
            # Similarly windowed metrics are not relevant for single predictions.
            window_metric_providers=[],
            global_metric_providers=self.evaluation_metrics,
        )

        evaluation_result = pipeline.run_for_subset(
            filtering=self.config[0].max_horizon,
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


__all__ = ["EnsembleForecastingModel", "ModelFitResult", "restore_target"]
