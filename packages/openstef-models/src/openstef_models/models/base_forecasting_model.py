# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Abstract base class for forecasting models.

Provides shared fields and evaluation logic used by both single-model
(ForecastingModel) and ensemble (EnsembleForecastingModel) implementations.
"""

import logging
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, override

import pandas as pd
from pydantic import Field, PrivateAttr

from openstef_beam.evaluation import EvaluationConfig, EvaluationPipeline, SubsetMetric
from openstef_beam.evaluation.metric_providers import MetricProvider, ObservedProbabilityProvider, R2Provider
from openstef_core.base_model import BaseModel
from openstef_core.datasets import ForecastDataset, TimeSeriesDataset
from openstef_core.mixins import Predictor, TransformPipeline
from openstef_core.mixins.forecaster import ForecasterConfig
from openstef_models.utils.data_split import DataSplitter


# TODO: Move to openstef-core?
class BaseForecastingModel(BaseModel, Predictor[TimeSeriesDataset, ForecastDataset]):
    """Abstract base for forecasting models."""

    # Shared model components
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
    @abstractmethod
    def scoring_config(self) -> ForecasterConfig:
        """Return the forecaster config used for evaluation metrics.

        For a single-model pipeline this is the forecaster's own config.
        For an ensemble it is typically the first (or canonical) base-forecaster config.
        """

    @abstractmethod
    @override
    def fit(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
    ) -> Any:
        """Train the forecasting model on the provided dataset.

        Args:
            data: Historical time series data with features and target values.
            data_val: Optional validation data.
            data_test: Optional test data.

        Returns:
            Fit result containing training details and metrics.
        """

    @abstractmethod
    @override
    def predict(self, data: TimeSeriesDataset, forecast_start: datetime | None = None) -> ForecastDataset:
        """Generate forecasts for the input data.

        Args:
            data: Input dataset for generating forecasts.
            forecast_start: Optional start time for forecasts.

        Returns:
            Generated forecast dataset.
        """

    def score(self, data: TimeSeriesDataset) -> SubsetMetric:
        """Evaluate model performance on the provided dataset.

        Generates predictions for the dataset and calculates evaluation metrics
        by comparing against ground truth values. Uses the configured evaluation
        metrics to assess forecast quality at the maximum forecast horizon.

        Args:
            data: Time series dataset containing both features and target values
                for evaluation.

        Returns:
            Evaluation metrics including configured providers (e.g., R², observed
            probability) computed at the maximum forecast horizon.
        """
        prediction = self.predict(data=data)
        return self._calculate_score(prediction=prediction)

    def _calculate_score(self, prediction: ForecastDataset) -> SubsetMetric:
        if prediction.target_series is None:
            raise ValueError("Prediction dataset must contain target series for scoring.")

        # Drop NaN targets for metric calculation
        prediction = prediction.pipe_pandas(pd.DataFrame.dropna, subset=[self.target_column])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        pipeline = EvaluationPipeline(
            config=EvaluationConfig(available_ats=[], lead_times=[self.scoring_config.max_horizon]),
            quantiles=self.scoring_config.quantiles,
            window_metric_providers=[],
            global_metric_providers=self.evaluation_metrics,
        )

        evaluation_result = pipeline.run_for_subset(
            filtering=self.scoring_config.max_horizon,
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


__all__ = ["BaseForecastingModel"]
