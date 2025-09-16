# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Callback interface for monitoring and extending forecasting workflows.

Provides hooks into key workflow stages to enable custom logging, metrics collection,
model validation, and integration with external monitoring systems. Callbacks follow
the observer pattern and are called at specific workflow lifecycle events.
"""

from typing import TYPE_CHECKING

from openstef_core.datasets import TimeSeriesDataset, VersionedTimeSeriesDataset
from openstef_core.datasets.validated_datasets import ForecastDataset

if TYPE_CHECKING:
    from openstef_models.workflows.forecasting_workflow import ForecastingWorkflow


class ForecastingCallback:
    """Base callback interface for monitoring forecasting workflow lifecycle events.

    Provides hooks at key stages of the forecasting process to enable custom
    functionality such as logging, metrics collection, model validation,
    data preprocessing, and integration with monitoring systems.

    All methods have default no-op implementations, so subclasses only need
    to override the specific events they care about.

    Example:
        Creating a logging callback:

        >>> class LoggingCallback(ForecastingCallback):
        ...     def on_before_fit(self, pipeline, dataset):
        ...         print(f"Starting training with {len(dataset.data)} samples")
        ...
        ...     def on_after_predict(self, pipeline, dataset, forecasts):
        ...         print(f"Generated {len(forecasts.data)} forecasts")
        >>>
        >>> callback = LoggingCallback()
        >>> workflow = ForecastingWorkflow(model, callbacks=callback) # doctest: +SKIP
    """

    def on_before_fit(self, pipeline: "ForecastingWorkflow", dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        """Called before model fitting begins.

        Use this hook for pre-training validation, data preprocessing,
        or setting up training monitoring.

        Args:
            pipeline: The forecasting workflow performing the fit.
            dataset: Training dataset being used for fitting.
        """

    def on_after_fit(self, pipeline: "ForecastingWorkflow", dataset: VersionedTimeSeriesDataset | TimeSeriesDataset):
        """Called after model fitting completes successfully.

        Use this hook for post-training validation, model evaluation,
        saving training metrics, or triggering downstream processes.

        Args:
            pipeline: The forecasting workflow that completed fitting.
            dataset: Training dataset that was used for fitting.
        """

    def on_before_predict(
        self, pipeline: "ForecastingWorkflow", dataset: VersionedTimeSeriesDataset | TimeSeriesDataset
    ):
        """Called before prediction generation begins.

        Use this hook for input data validation, prediction setup,
        or logging prediction requests.

        Args:
            pipeline: The forecasting workflow performing the prediction.
            dataset: Input dataset being used for prediction.
        """

    def on_after_predict(
        self,
        pipeline: "ForecastingWorkflow",
        dataset: VersionedTimeSeriesDataset | TimeSeriesDataset,
        forecasts: ForecastDataset,
    ):
        """Called after prediction generation completes successfully.

        Use this hook for forecast validation, saving predictions,
        calculating metrics, or triggering downstream processes.

        Args:
            pipeline: The forecasting workflow that completed prediction.
            dataset: Input dataset that was used for prediction.
            forecasts: Generated forecast results.
        """
