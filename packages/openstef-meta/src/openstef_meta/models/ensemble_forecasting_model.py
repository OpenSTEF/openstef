# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Ensemble forecasting model combining multiple base forecasters.

Orchestrates parallel base forecasters whose predictions are aggregated by a
``ForecastCombiner``.  Extends ``BaseForecastingModel`` as a sibling of
``ForecastingModel``.
"""

import logging
from datetime import datetime
from functools import partial
from typing import Self, cast, override

import pandas as pd
from pydantic import Field, PrivateAttr, model_validator

from openstef_core.datasets import (
    ForecastDataset,
    ForecastInputDataset,
    TimeSeriesDataset,
)
from openstef_core.datasets.timeseries_dataset import validate_horizons_present
from openstef_core.datasets.validated_datasets import EnsembleForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import HyperParams, TransformPipeline
from openstef_core.types import LeadTime, Quantile
from openstef_meta.models.forecast_combiners.forecast_combiner import ForecastCombiner
from openstef_models.explainability.mixins import ExplainableForecaster
from openstef_models.models.forecasting.forecaster import Forecaster
from openstef_models.models.forecasting_model import BaseForecastingModel, ModelFitResult, restore_target

logger = logging.getLogger(__name__)


class EnsembleModelFitResult(ModelFitResult):
    """Fit result for EnsembleForecastingModel.

    Extends ModelFitResult with per-forecaster details. The base class fields
    (input_dataset, metrics_*, etc.) represent the combiner's fit results.
    """

    forecaster_fit_results: dict[str, ModelFitResult] = Field(description="ModelFitResult for each base forecaster")

    @override
    def metrics_to_flat_dict(self) -> dict[str, float]:
        result = super().metrics_to_flat_dict()
        for name, child in self.forecaster_fit_results.items():
            result.update({f"{name}_{k}": v for k, v in child.metrics_to_flat_dict().items()})
        return result

    @property
    @override
    def component_fit_results(self) -> dict[str, ModelFitResult]:
        return self.forecaster_fit_results


class EnsembleForecastingModel(BaseForecastingModel):
    """Ensemble forecasting pipeline: common preprocessing -> N forecasters -> combiner.

    Runs multiple base forecasters in parallel, aggregates their predictions via a
    ``ForecastCombiner``, and applies shared postprocessing.  Extends
    ``BaseForecastingModel`` as a sibling of ``ForecastingModel`` — not a subclass.

    The ``preprocessing`` field (inherited from base) holds the **common preprocessing**
    shared across all base forecasters.  ``model_specific_preprocessing`` adds
    per-forecaster transforms on top.

    Invariants:
        - fit() must be called before predict()
        - All forecaster horizons must be present in the input data

    Important:
        The ``cutoff_history`` parameter is crucial when using lag-based features.
        Set it to exclude incomplete rows from training (e.g. ``timedelta(days=14)``
        for a lag-14 transform).

    Example:
        >>> from openstef_models.models.forecasting.constant_median_forecaster import (
        ...     ConstantMedianForecaster,
        ... )
        >>> from openstef_meta.models.forecast_combiners.learned_weights_combiner import WeightsCombiner
        >>> from openstef_core.types import LeadTime
        >>> from datetime import timedelta
        >>>
        >>> forecaster_1 = ConstantMedianForecaster(
        ...     horizons=[LeadTime.from_string("PT36H")]
        ... )
        >>> forecaster_2 = ConstantMedianForecaster(
        ...     horizons=[LeadTime.from_string("PT36H")]
        ... )
        >>> combiner = WeightsCombiner(
        ...     horizons=[LeadTime.from_string("PT36H")],
        ... )
        >>> model = EnsembleForecastingModel(
        ...     forecasters={"constant_median": forecaster_1, "constant_median_2": forecaster_2},
        ...     combiner=combiner,
        ...     cutoff_history=timedelta(days=14),
        ... )
        >>> model.fit(training_data)  # doctest: +SKIP
        >>> forecasts = model.predict(new_data)  # doctest: +SKIP
    """

    forecasters: dict[str, Forecaster] = Field(
        default=...,
        description="Named base forecasters whose predictions are combined.",
        exclude=True,
    )

    combiner: ForecastCombiner = Field(
        default=...,
        description="Combiner that aggregates base forecaster predictions.",
        exclude=True,
    )

    model_specific_preprocessing: dict[str, TransformPipeline[TimeSeriesDataset]] = Field(
        default_factory=dict,
        description="Per-forecaster preprocessing pipelines applied after common preprocessing.",
        exclude=True,
    )

    combiner_preprocessing: TransformPipeline[TimeSeriesDataset] = Field(
        default_factory=TransformPipeline[TimeSeriesDataset],
        description="Feature engineering for the forecast combiner.",
        exclude=True,
    )

    model_specific_postprocessing: TransformPipeline[ForecastDataset] = Field(
        default_factory=TransformPipeline[ForecastDataset],
        description="Per-forecaster postprocessing applied before the combiner sees predictions.",
        exclude=True,
    )

    combiner_postprocessing: TransformPipeline[ForecastDataset] = Field(
        default_factory=TransformPipeline[ForecastDataset],
        description="Combiner-specific postprocessing applied after shared postprocessing.",
        exclude=True,
    )

    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))

    @model_validator(mode="after")
    def _validate_horizons_consistent(self) -> Self:
        """All forecasters and the combiner must share the same horizons list.

        Returns:
            Validated model instance.

        Raises:
            ValueError: If forecasters dict is empty or any forecaster's horizons differ from the combiner's.
        """
        if not self.forecasters:
            msg = "At least one forecaster is required."
            raise ValueError(msg)

        expected = sorted(self.combiner.horizons)
        for name, forecaster in self.forecasters.items():
            if sorted(forecaster.horizons) != expected:
                msg = (
                    f"Forecaster '{name}' horizons {forecaster.horizons} "
                    f"do not match combiner horizons {self.combiner.horizons}"
                )
                raise ValueError(msg)
        return self

    @property
    def forecaster_configs(self) -> dict[str, Forecaster]:
        """Configuration of each base forecaster, keyed by name."""
        return dict(self.forecasters)

    @property
    @override
    def quantiles(self) -> list[Quantile]:
        return self.combiner.quantiles

    @property
    @override
    def max_horizon(self) -> LeadTime:
        return self.combiner.max_horizon

    @property
    @override
    def hyperparams(self) -> HyperParams:
        return self.combiner.hparams

    @property
    @override
    def is_fitted(self) -> bool:
        return all(f.is_fitted for f in self.forecasters.values()) and self.combiner.is_fitted

    @property
    @override
    def component_hyperparams(self) -> dict[str, HyperParams]:
        return {name: f.hparams for name, f in self.forecasters.items()}

    @override
    def get_explainable_components(self) -> dict[str, ExplainableForecaster]:
        components: dict[str, ExplainableForecaster] = {
            name: forecaster
            for name, forecaster in self.forecasters.items()
            if isinstance(forecaster, ExplainableForecaster)
        }
        # ForecastCombiner is always ExplainableForecaster, but skip if importances are empty
        if not self.combiner.feature_importances.empty:
            components["combiner"] = self.combiner
        return components

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
        """Train all base forecasters and then the combiner.

        Args:
            data: Historical time series data with features and target values.
            data_val: Optional validation data. If provided, splitters are ignored for validation.
            data_test: Optional test data. If provided, splitters are ignored for test.

        Returns:
            FitResult containing training details and metrics.
        """
        # Phase 1: fit each base forecaster and collect their in-sample predictions
        train_ensemble, val_ensemble, test_ensemble, forecaster_fit_results = self._fit_forecasters(
            data=data,
            data_val=data_val,
            data_test=data_test,
        )

        # Phase 2: fit the combiner on base forecasters' in-sample predictions
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
            **combiner_fit_result.model_dump(),
        )

    @staticmethod
    def _combine_datasets(
        data: ForecastInputDataset, additional_features: ForecastInputDataset
    ) -> ForecastInputDataset:
        """Combine Forecaster learner predictions with additional features for ForecastCombiner input.

        Args:
            data: ForecastInputDataset containing base Forecaster predictions.
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
        # Returns None when no combiner preprocessing is configured, signalling the combiner
        # should work without additional features.
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
        # Fits combiner preprocessing on train data and transforms all splits.
        # Returns (None, None, None) when no combiner preprocessing is configured.
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
        # Fits common + per-forecaster preprocessing, trains each forecaster,
        # and bundles their in-sample predictions into EnsembleForecastDatasets.
        predictions_train: dict[str, ForecastDataset] = {}
        predictions_val: dict[str, ForecastDataset | None] = {}
        predictions_test: dict[str, ForecastDataset | None] = {}
        results: dict[str, ModelFitResult] = {}

        # Fit the feature engineering transforms
        self.preprocessing.fit(data=data)
        data_transformed = self.preprocessing.transform(data=data)
        # Fit per-forecaster transforms on the common-preprocessed output (not raw data)
        for name in self.model_specific_preprocessing:
            self.model_specific_preprocessing[name].fit(data=data_transformed)
        logger.debug("Completed fitting preprocessing pipelines.")

        # Fit the forecasters
        for name in self.forecasters:
            logger.debug("Fitting Forecaster '%s'.", name)
            predictions_train[name], predictions_val[name], predictions_test[name], results[name] = (
                self._fit_forecaster(
                    data=data,
                    data_val=data_val,
                    data_test=data_test,
                    forecaster_name=name,
                )
            )

        # Attach original (unsplit) target so the combiner can compute loss across all timesteps
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
        validate_horizons_present(data, forecaster.horizons)

        # Transform and split input data
        input_data_train = self.prepare_forecaster_input(data=data, forecaster_name=forecaster_name)
        input_data_val = (
            self.prepare_forecaster_input(data=data_val, forecaster_name=forecaster_name) if data_val else None
        )
        input_data_test = (
            self.prepare_forecaster_input(data=data_test, forecaster_name=forecaster_name) if data_test else None
        )

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
        logger.debug("Predicting forecaster '%s'.", forecaster_name)
        prediction_raw = self.forecasters[forecaster_name].predict(data=input_data)
        prediction = restore_target(
            dataset=prediction_raw, original_dataset=input_data, target_column=self.target_column
        )
        prediction = self.model_specific_postprocessing.transform(prediction)
        return self.postprocessing.transform(prediction)

    def _predict_forecasters(
        self,
        data: TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> EnsembleForecastDataset:
        predictions: dict[str, ForecastDataset] = {}
        for name in self.forecasters:
            logger.debug("Generating predictions for forecaster '%s'.", name)
            input_data = self.prepare_forecaster_input(data=data, forecast_start=forecast_start, forecaster_name=name)
            predictions[name] = self._predict_forecaster(
                input_data=input_data,
                forecaster_name=name,
            )

        return EnsembleForecastDataset.from_forecast_datasets(predictions, target_series=data.data[self.target_column])

    def prepare_forecaster_input(
        self,
        data: TimeSeriesDataset,
        forecaster_name: str = "",
        forecast_start: datetime | None = None,
    ) -> ForecastInputDataset:
        """Prepare input data for a specific base forecaster.

        Applies common preprocessing, then model-specific preprocessing, restores
        the target column, and trims history via the shared base ``prepare_input``.

        Args:
            data: Raw time series dataset.
            forecaster_name: Which forecaster to prepare data for.
            forecast_start: Optional forecast start time override.

        Returns:
            Processed forecast input dataset ready for the named forecaster.
        """
        logger.debug("Preparing input data for forecaster '%s'.", forecaster_name)
        # Apply model-specific preprocessing on top of the common pipeline
        if forecaster_name in self.model_specific_preprocessing:
            logger.debug("Applying model-specific preprocessing for forecaster '%s'.", forecaster_name)
            preprocessed = self.preprocessing.transform(data=data)
            preprocessed = self.model_specific_preprocessing[forecaster_name].transform(data=preprocessed)
            preprocessed = restore_target(dataset=preprocessed, original_dataset=data, target_column=self.target_column)
            # Apply cutoff and create ForecastInputDataset
            input_data_start = cast("pd.Series[pd.Timestamp]", preprocessed.index).min().to_pydatetime()
            input_data_cutoff = input_data_start + self.cutoff_history
            if forecast_start is not None and forecast_start < input_data_cutoff:
                input_data_cutoff = forecast_start
                self._logger.warning(
                    "Forecast start %s is before input data start + cutoff history %s. Using forecast start as cutoff.",
                    forecast_start,
                    input_data_cutoff,
                )
            preprocessed = preprocessed.filter_by_range(start=input_data_cutoff)

            return ForecastInputDataset.from_timeseries(
                dataset=preprocessed,
                target_column=self.target_column,
                forecast_start=forecast_start,
            )

        # No model-specific preprocessing — delegate entirely to shared base method
        return self.prepare_input(data=data, forecast_start=forecast_start)

    def _predict_transform_combiner(
        self, ensemble_dataset: EnsembleForecastDataset, original_data: TimeSeriesDataset
    ) -> ForecastDataset:
        logger.debug("Predicting combiner.")
        features = self._transform_combiner_data(data=original_data)

        return self._predict_combiner(ensemble_dataset, features)

    def _predict_combiner(
        self,
        ensemble_dataset: EnsembleForecastDataset,
        features: ForecastInputDataset | None,
    ) -> ForecastDataset:
        logger.debug("Predicting combiner.")
        prediction_raw = self.combiner.predict(ensemble_dataset, additional_features=features)
        prediction = restore_target(
            dataset=prediction_raw, original_dataset=ensemble_dataset, target_column=self.target_column
        )
        prediction = self.combiner_postprocessing.transform(prediction)
        return self.postprocessing.transform(prediction)

    def _fit_combiner(
        self,
        data: TimeSeriesDataset,
        train_ensemble_dataset: EnsembleForecastDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
        val_ensemble_dataset: EnsembleForecastDataset | None = None,
        test_ensemble_dataset: EnsembleForecastDataset | None = None,
    ) -> ModelFitResult:
        # Prepare additional features for the combiner (e.g. sample weights) — split separately from ensemble data
        features_train, features_val, features_test = self._fit_prepare_combiner_data(
            data=data, data_val=data_val, data_test=data_test
        )

        logger.debug("Fitting combiner.")
        self.combiner.fit(
            data=train_ensemble_dataset, data_val=val_ensemble_dataset, additional_features=features_train
        )

        # Fit combiner postprocessing on training predictions
        prediction_raw = self.combiner.predict(train_ensemble_dataset, additional_features=features_train)
        prediction_raw = restore_target(
            dataset=prediction_raw, original_dataset=train_ensemble_dataset, target_column=self.target_column
        )
        self.combiner_postprocessing.fit_transform(prediction_raw)

        prediction_train = self._predict_combiner(train_ensemble_dataset, features=features_train)
        metrics_train = self._calculate_score(prediction=prediction_train)

        if val_ensemble_dataset is not None:
            prediction_val = self._predict_combiner(val_ensemble_dataset, features=features_val)
            metrics_val = self._calculate_score(prediction=prediction_val)
        else:
            prediction_val = None
            metrics_val = None

        if test_ensemble_dataset is not None:
            prediction_test = self._predict_combiner(test_ensemble_dataset, features=features_test)
            metrics_test = self._calculate_score(prediction=prediction_test)
        else:
            prediction_test = None
            metrics_test = None

        return ModelFitResult(
            input_dataset=train_ensemble_dataset,
            # ModelFitResult expects ForecastInputDataset; use first quantile as a representative slice
            input_data_train=train_ensemble_dataset.get_base_predictions_for_quantile(quantile=self.quantiles[0]),
            input_data_val=val_ensemble_dataset.get_base_predictions_for_quantile(quantile=self.quantiles[0])
            if val_ensemble_dataset
            else None,
            input_data_test=test_ensemble_dataset.get_base_predictions_for_quantile(quantile=self.quantiles[0])
            if test_ensemble_dataset
            else None,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            metrics_test=metrics_test,
            metrics_full=metrics_train,
        )

    def _predict_contributions_combiner(
        self, ensemble_dataset: EnsembleForecastDataset, original_data: TimeSeriesDataset
    ) -> TimeSeriesDataset:
        features = self._transform_combiner_data(data=original_data)
        return self.combiner.predict_contributions(ensemble_dataset, additional_features=features)

    @override
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
        return self._predict_transform_combiner(
            ensemble_dataset=ensemble_predictions,
            original_data=data,
        )

    @override
    def predict_contributions(
        self,
        data: TimeSeriesDataset,
        forecast_start: datetime | None = None,
    ) -> TimeSeriesDataset:
        """Compute per-model contributions for the ensemble prediction.

        Args:
            data: Input time series dataset.
            forecast_start: Optional start time for forecasts.

        Returns:
            TimeSeriesDataset where each column is a base model's contribution.

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


__all__ = ["EnsembleForecastingModel", "EnsembleModelFitResult", "ModelFitResult"]
