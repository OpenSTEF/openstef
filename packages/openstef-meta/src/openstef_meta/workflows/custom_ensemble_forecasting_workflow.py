# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level workflow orchestration for ensemble forecasting operations.

Provides a complete ensemble forecasting workflow that combines model management,
callback execution, and optional model persistence. Acts as the main entry point
for production ensemble forecasting systems.
"""

import logging
from datetime import datetime

from pydantic import Field, PrivateAttr

from openstef_core.base_model import BaseModel
from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset
from openstef_core.exceptions import NotFittedError, SkipFitting
from openstef_meta.models.ensemble_forecasting_model import (
    EnsembleForecastingModel,
    EnsembleModelFitResult,
)
from openstef_models.mixins import ModelIdentifier, PredictorCallback
from openstef_models.mixins.callbacks import WorkflowContext


class EnsembleForecastingCallback(
    PredictorCallback[
        "CustomEnsembleForecastingWorkflow",
        VersionedTimeSeriesDataset | TimeSeriesDataset,
        EnsembleModelFitResult,
        ForecastDataset,
    ]
):
    """Callback interface for monitoring ensemble forecasting workflow lifecycle events.

    Similar to ForecastingCallback but parameterized with EnsembleModelFitResult
    instead of ModelFitResult, giving callbacks access to the full ensemble fit
    result including per-forecaster and combiner results.

    All methods have default no-op implementations, so subclasses only need
    to override the specific events they care about.
    """


class CustomEnsembleForecastingWorkflow(BaseModel):
    """Complete ensemble forecasting workflow with model management and lifecycle hooks.

    Orchestrates the full ensemble forecasting process by combining an
    EnsembleForecastingModel with callback execution. Provides the main
    interface for production ensemble forecasting systems.

    Invariants:
        - Callbacks are executed at appropriate lifecycle stages
        - Model fitting returns EnsembleModelFitResult with per-forecaster details
        - Prediction delegates to the underlying EnsembleForecastingModel
    """

    model: EnsembleForecastingModel = Field(description="The ensemble forecasting model to use.")
    callbacks: list[EnsembleForecastingCallback] = Field(
        default_factory=list[EnsembleForecastingCallback],
        description="List of callbacks to execute during workflow events.",
    )
    model_id: ModelIdentifier = Field(...)
    run_name: str | None = Field(default=None, description="Optional name for this workflow run.")
    experiment_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for experiment tracking.",
    )

    _logger: logging.Logger = PrivateAttr(default_factory=lambda: logging.getLogger(__name__))

    def fit(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
        data_test: TimeSeriesDataset | None = None,
    ) -> EnsembleModelFitResult | None:
        """Train the ensemble forecasting model with callback execution.

        Executes the complete training workflow including pre-fit callbacks,
        model training, and post-fit callbacks. Returns the full ensemble
        fit result with per-forecaster and combiner details.

        Args:
            data: Training dataset for the forecasting model.
            data_val: Optional validation dataset for model tuning.
            data_test: Optional test dataset for final evaluation.

        Returns:
            EnsembleModelFitResult containing training metrics for each
            base forecaster and the combiner, or None if fitting was skipped.
        """
        result: EnsembleModelFitResult | None = None
        context: WorkflowContext[CustomEnsembleForecastingWorkflow] = WorkflowContext(workflow=self)

        try:
            for callback in self.callbacks:
                callback.on_fit_start(context=context, data=data)

            result = self.model.fit(data=data, data_val=data_val, data_test=data_test)

            for callback in self.callbacks:
                callback.on_fit_end(context=context, result=result)
        except SkipFitting as e:
            self._logger.info("Skipping model fitting: %s", e)
            result = None

        return result

    def predict(self, data: TimeSeriesDataset, forecast_start: datetime | None = None) -> ForecastDataset:
        """Generate forecasts with callback execution.

        Executes the complete prediction workflow including pre-prediction callbacks,
        model prediction, and post-prediction callbacks.

        Args:
            data: Input dataset for generating forecasts.
            forecast_start: Optional start time for forecasts.

        Returns:
            Generated forecast dataset.

        Raises:
            NotFittedError: If the underlying model hasn't been trained.
        """
        context: WorkflowContext[CustomEnsembleForecastingWorkflow] = WorkflowContext(workflow=self)

        for callback in self.callbacks:
            callback.on_predict_start(context=context, data=data)

        if not self.model.is_fitted:
            raise NotFittedError(type(self.model).__name__)

        forecasts = self.model.predict(data=data, forecast_start=forecast_start)

        for callback in self.callbacks:
            callback.on_predict_end(context=context, data=data, result=forecasts)

        return forecasts


__all__ = ["CustomEnsembleForecastingWorkflow", "EnsembleForecastingCallback"]
