# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Callback interface for monitoring and extending forecasting workflows.

Provides hooks into key workflow stages to enable custom logging, metrics collection,
model validation, and integration with external monitoring systems. Callbacks follow
the observer pattern and are called at specific workflow lifecycle events.
"""

from typing import Any

from pydantic import Field

from openstef_core.base_model import BaseModel


class WorkflowContext[W](BaseModel):
    """Context object for workflow execution state.

    Holds the current workflow instance and provides storage for additional
    data that needs to be passed between workflow stages or shared with
    callbacks.
    """

    workflow: W = Field(description="The workflow instance being executed.")
    data: dict[str, Any] = Field(default_factory=dict, description="Dictionary for storing arbitrary additional data.")


class PredictorCallback[W, I, O]:
    """Base callback interface for monitoring predictor workflow lifecycle events.

    Provides hooks at key stages of the prediction process to enable custom
    functionality such as logging, metrics collection, model validation,
    data preprocessing, and integration with monitoring systems.

    All methods have default no-op implementations, so subclasses only need
    to override the specific events they care about.
    """

    def on_fit_start(self, context: WorkflowContext[W], data: I):
        """Called before model fitting begins.

        Use this hook for pre-training validation, data preprocessing,
        or setting up training monitoring.

        Args:
            context: The workflow context performing the fitting.
            data: Training dataset being used for fitting.
        """

    def on_fit_end(self, context: WorkflowContext[W], data: I):
        """Called after model fitting completes successfully.

        Use this hook for post-training validation, model evaluation,
        saving training metrics, or triggering downstream processes.

        Args:
            context: The workflow context that completed fitting.
            data: Training dataset that was used for fitting.
        """

    def on_predict_start(self, context: WorkflowContext[W], data: I):
        """Called before prediction generation begins.

        Use this hook for input data validation, prediction setup,
        or logging prediction requests.

        Args:
            context: The workflow context performing the prediction.
            data: Input dataset being used for prediction.
        """

    def on_predict_end(
        self,
        context: WorkflowContext[W],
        data: I,
        forecasts: O,
    ):
        """Called after prediction generation completes successfully.

        Use this hook for forecast validation, saving predictions,
        calculating metrics, or triggering downstream processes.

        Args:
            context: The workflow context that completed prediction.
            data: Input dataset that was used for prediction.
            forecasts: Generated forecast results.
        """


__all__ = ["PredictorCallback", "WorkflowContext"]
