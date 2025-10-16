# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""High-level workflow orchestration for component splitting operations.

Provides a complete component splitting workflow that combines model management,
callback execution, and optional model persistence. Acts as the main
entry point for production component splitting systems.
"""

from pydantic import Field

from openstef_core.base_model import BaseModel
from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.mixins import ModelIdentifier, PredictorCallback
from openstef_models.mixins.callbacks import WorkflowContext
from openstef_models.models import ComponentSplittingModel


class ComponentSplitCallback(
    PredictorCallback["CustomComponentSplitWorkflow", TimeSeriesDataset, None, EnergyComponentDataset]
):
    """Base callback interface for monitoring component splitting workflow lifecycle events.

    Provides hooks at key stages of the component splitting process to enable custom
    functionality such as logging, metrics collection, model validation,
    data preprocessing, and integration with monitoring systems.

    All methods have default no-op implementations, so subclasses only need
    to override the specific events they care about.

    Example:
        Creating a logging callback:

        >>> class LoggingCallback(ComponentSplitCallback):
        ...     def on_fit_start(self, pipeline, dataset):
        ...         print(f"Starting training with {len(dataset.data)} samples")
        ...
        ...     def on_predict_end(self, pipeline, dataset, forecasts):
        ...         print(f"Generated {len(forecasts.data)} forecasts")
        >>>
        >>> callback = LoggingCallback()
        >>> workflow = ComponentSplitWorkflow(model, callbacks=callback) # doctest: +SKIP
    """


class CustomComponentSplitWorkflow(BaseModel):
    """Complete component splitting workflow with model management and lifecycle hooks.

    Orchestrates the full component splitting process by combining a ComponentSplittingModel
    with callback execution and optional model persistence. Provides the main
    interface for production component splitting systems where models need to be
    trained, saved, loaded, and used for prediction with monitoring.

    Invariants:
        - Callbacks are executed at appropriate lifecycle stages
        - Model fitting and prediction delegate to the underlying ComponentSplittingModel
        - Storage operations (if configured) maintain model persistence

    Example:
        Basic workflow with callbacks:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from datetime import timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.models import ComponentSplittingModel
        >>>
        >>> # Create sample data
        >>> dataset = TimeSeriesDataset(
        ...     data=pd.DataFrame({
        ...         "load": np.random.default_rng(42).standard_normal(size=48),
        ...     }, index=pd.date_range("2025-01-01", periods=48, freq="h")),
        ...     sample_interval=timedelta(hours=1),
        ... )
        >>>
        >>> # Create model and workflow
        >>> model = ComponentSplittingModel(...)  # doctest: +SKIP
        >>>
        >>> class LoggingCallback(ComponentSplitCallback):
        ...     def on_fit_end(self, workflow, data):
        ...         print("Model training completed")
        >>>
        >>> workflow = CustomComponentSplitWorkflow(
        ...     model=model, model_id="my_model", callbacks=LoggingCallback()
        ... ) # doctest: +SKIP
        >>> workflow.fit(dataset) # doctest: +SKIP
        Model training completed
        >>> components = workflow.predict(dataset) # doctest: +SKIP

        Loading from storage with fallback:

        >>> workflow = ComponentSplitWorkflow.from_storage(
        ...     model_id="production_model_v1",
        ...     storage=my_storage,
        ...     default_model_factory=lambda: create_default_model()
        ... ) # doctest: +SKIP
    """

    model: ComponentSplittingModel = Field(description="The component splitting model to use.")
    callbacks: ComponentSplitCallback = Field(default_factory=ComponentSplitCallback)
    model_id: ModelIdentifier = Field(...)

    def fit(
        self,
        data: TimeSeriesDataset,
        data_val: TimeSeriesDataset | None = None,
    ) -> None:
        """Train the component splitting model with callback execution.

        Executes the complete training workflow including pre-fit callbacks,
        model training, and post-fit callbacks.

        Args:
            data: Training dataset for the component splitting model.
            data_val: Optional validation dataset for monitoring during training.
        """
        context: WorkflowContext[CustomComponentSplitWorkflow] = WorkflowContext(workflow=self)

        self.callbacks.on_fit_start(context=context, data=data)

        result = self.model.fit(data=data, data_val=data_val)

        self.callbacks.on_fit_end(context=context, result=result)

    def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
        """Generate component predictions with callback execution.

        Executes the complete prediction workflow including pre-prediction callbacks,
        model prediction, and post-prediction callbacks.

        Args:
            data: Input dataset for generating component predictions.

        Returns:
            Generated energy component dataset.

        Raises:
            NotFittedError: If the underlying model hasn't been trained.
        """
        if not self.model.is_fitted:
            raise NotFittedError(type(self.model).__name__)

        context: WorkflowContext[CustomComponentSplitWorkflow] = WorkflowContext(workflow=self)

        self.callbacks.on_predict_start(context=context, data=data)

        prediction = self.model.predict(data=data)

        self.callbacks.on_predict_end(context=context, data=data, result=prediction)

        return prediction
