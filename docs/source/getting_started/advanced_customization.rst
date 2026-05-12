Advanced Customization
======================

This page covers the main extension points in OpenSTEF for power users who need to go beyond the built-in presets. You will learn how to create custom transforms for feature engineering, build custom preprocessing and postprocessing pipelines, implement workflow callbacks for monitoring, and compose custom forecasting workflows.

If you haven't yet run your first forecast, start with the :doc:`first_forecast` tutorial.

.. mermaid:: /diagrams/getting_started/advanced_customization_diagram_1.mmd

Custom Transforms
-----------------

The transform system is OpenSTEF's primary mechanism for feature engineering and data preparation. All transforms implement a common interface with ``fit``, ``transform``, and ``fit_transform`` methods.

Creating a Custom TimeSeriesTransform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a custom feature transform, subclass ``TimeSeriesTransform`` and implement the ``transform`` and ``features_added`` methods:

.. code-block:: python

   from openstef_core.datasets import TimeSeriesDataset
   from openstef_models.transforms.energy_domain import TimeSeriesTransform


   class TemperatureGradientAdder(TimeSeriesTransform):
       """Adds temperature rate-of-change as a feature."""

       @property
       def is_fitted(self) -> bool:
           # Stateless transform - always fitted
           return True

       def fit(self, data: TimeSeriesDataset) -> None:
           pass

       def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
           df = data.data.copy()
           if "temperature" in df.columns:
               df["temperature_gradient"] = df["temperature"].diff()
           return TimeSeriesDataset(df, data.sample_interval)

       def features_added(self) -> list[str]:
           return ["temperature_gradient"]

The ``features_added`` method declares which columns the transform introduces, enabling pipeline introspection and validation.

Stateful Transforms
^^^^^^^^^^^^^^^^^^^

For transforms that learn parameters from training data, override ``is_fitted`` and ``fit``:

.. code-block:: python

   from openstef_core.datasets import TimeSeriesDataset
   from openstef_models.transforms.energy_domain import TimeSeriesTransform


   class MinMaxScaler(TimeSeriesTransform):
       """Scales the target column to [0, 1] range."""

       scale_factor: float | None = None

       @property
       def is_fitted(self) -> bool:
           return self.scale_factor is not None

       def fit(self, data: TimeSeriesDataset) -> None:
           self.scale_factor = data.data.max().max()

       def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
           if not self.is_fitted:
               raise RuntimeError("Transform not fitted. Call fit() first.")
           scaled_data = data.data / self.scale_factor
           return TimeSeriesDataset(scaled_data, data.sample_interval)

       def features_added(self) -> list[str]:
           return []

Stateful transforms are fitted during training and their learned parameters are persisted with the model.

Custom Pipelines
----------------

The ``TransformPipeline`` chains multiple transforms into a sequential processing flow. Each transform receives the output of the previous one.

.. code-block:: python

   from openstef_core.datasets import TimeSeriesDataset
   from openstef_models.transforms import TransformPipeline
   from openstef_models.transforms.energy_domain.wind_power_feature_adder import WindPowerFeatureAdder

   preprocessing = TransformPipeline[TimeSeriesDataset](
       transforms=[
           WindPowerFeatureAdder(),
           TemperatureGradientAdder(),
       ]
   )

   # Fit and transform in one step
   processed_data = preprocessing.fit_transform(raw_data)

   # Or separately for train/test splits
   preprocessing.fit(train_data)
   processed_train = preprocessing.transform(train_data)
   processed_test = preprocessing.transform(test_data)

An empty pipeline acts as a no-op, which is useful for conditional processing:

.. code-block:: python

   # No-op pipeline
   passthrough = TransformPipeline[TimeSeriesDataset](transforms=[])

Built-in Transform Domains
^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenSTEF organizes transforms into domain-specific subpackages:

- ``openstef_models.transforms.energy_domain`` — Energy-specific features (e.g., wind power curves)
- ``openstef_models.transforms.weather_domain`` — Weather-derived features
- ``openstef_models.transforms.time_domain`` — Temporal features (lags, calendar effects)
- ``openstef_models.transforms.general`` — General-purpose transforms
- ``openstef_models.transforms.validation`` — Data quality and validation transforms

Custom Forecasting Workflows
-----------------------------

The ``CustomForecastingWorkflow`` combines a model with preprocessing, postprocessing, and lifecycle callbacks into a complete forecasting system.

.. code-block:: python

   from openstef_models.workflows.custom_forecasting_workflow import CustomForecastingWorkflow
   from openstef_models.models import ForecastingModel
   from openstef_models.transforms import TransformPipeline

   workflow = CustomForecastingWorkflow(
       model=ForecastingModel(
           preprocessing=TransformPipeline(transforms=[
               WindPowerFeatureAdder(),
               TemperatureGradientAdder(),
           ]),
           forecaster=my_forecaster,
           postprocessing=TransformPipeline(transforms=[
               ConfidenceIntervalApplicator(quantiles=[0.1, 0.5, 0.9]),
           ]),
           target_column="load",
           data_splitter=my_splitter,
           cutoff_history=timedelta(days=365),
           evaluation_metrics=my_metrics,
           tags={"location": "substation_A"},
       ),
       model_id="custom-model-001",
       run_name="experiment-v1",
       callbacks=[],
   )

   # Train
   result = workflow.fit(train_data, data_val=val_data)

   # Predict
   forecasts = workflow.predict(new_data)

You can create a variant of an existing workflow with a different run name using ``with_run_name``:

.. code-block:: python

   experiment_b = workflow.with_run_name("experiment-v2")

Lifecycle Callbacks
-------------------

Callbacks let you hook into workflow lifecycle events without modifying core logic. This is the recommended pattern for logging, monitoring, model storage, and custom validation.

.. code-block:: python

   from openstef_models.workflows.custom_forecasting_workflow import ForecastingCallback


   class LoggingCallback(ForecastingCallback):
       """Logs training and prediction events."""

       def on_fit_start(self, pipeline, dataset):
           print(f"Starting training with {len(dataset.data)} samples")

       def on_fit_end(self, pipeline, dataset, result):
           print(f"Training complete. Metrics: {result}")

       def on_predict_start(self, pipeline, dataset):
           print(f"Generating forecast for {len(dataset.data)} time steps")

       def on_predict_end(self, pipeline, dataset, forecasts):
           print(f"Forecast generated: {len(forecasts.data)} predictions")

Register callbacks when constructing the workflow:

.. code-block:: python

   workflow = CustomForecastingWorkflow(
       model=my_model,
       model_id="custom-model-001",
       callbacks=[LoggingCallback()],
   )

OpenSTEF includes a built-in ``MLFlowStorageCallback`` for model persistence and selection:

.. code-block:: python

   from openstef_models.workflows.custom_forecasting_workflow import MLFlowStorageCallback

   mlflow_callback = MLFlowStorageCallback(
       storage=my_mlflow_storage,
       model_reuse_enable=True,
       model_reuse_max_age=timedelta(days=7),
       model_selection_enable=True,
       model_selection_metric="rmse",
       model_selection_old_model_penalty=0.05,
   )

.. mermaid:: /diagrams/getting_started/advanced_customization_diagram_2.mmd

Putting It All Together
-----------------------

Here is a complete example combining custom transforms, a pipeline, and callbacks into a production-ready workflow:

.. code-block:: python

   from datetime import timedelta

   from openstef_core.datasets import TimeSeriesDataset
   from openstef_models.models import ForecastingModel
   from openstef_models.transforms import TransformPipeline
   from openstef_models.workflows.custom_forecasting_workflow import (
       CustomForecastingWorkflow,
       ForecastingCallback,
   )


   # 1. Define custom transforms
   class PeakIndicatorAdder(TimeSeriesTransform):
       """Flags time steps during typical peak hours."""

       def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
           df = data.data.copy()
           df["is_peak_hour"] = df.index.hour.isin(range(17, 21)).astype(int)
           return TimeSeriesDataset(df, data.sample_interval)

       def features_added(self) -> list[str]:
           return ["is_peak_hour"]


   # 2. Define a monitoring callback
   class MetricsCallback(ForecastingCallback):
       def on_fit_end(self, pipeline, dataset, result):
           if result is not None:
               print(f"Model trained successfully: {result}")


   # 3. Assemble the workflow
   workflow = CustomForecastingWorkflow(
       model=ForecastingModel(
           preprocessing=TransformPipeline(transforms=[
               PeakIndicatorAdder(),
           ]),
           forecaster=my_forecaster,
           postprocessing=TransformPipeline(transforms=[]),
           target_column="load",
           cutoff_history=timedelta(days=180),
           tags={"use_case": "congestion_management"},
       ),
       model_id="peak-aware-model",
       callbacks=[MetricsCallback()],
   )

   # 4. Train and predict
   workflow.fit(train_data, data_val=val_data)
   forecasts = workflow.predict(live_data)

Summary of Extension Points
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Extension Point
     - Base Class
     - Use Case
   * - Feature engineering
     - ``TimeSeriesTransform``
     - Add domain-specific features
   * - Data preprocessing
     - ``TransformPipeline``
     - Chain transforms sequentially
   * - Workflow monitoring
     - ``ForecastingCallback``
     - Logging, storage, validation
   * - Full workflow
     - ``CustomForecastingWorkflow``
     - End-to-end custom forecasting

Next Steps
----------

- Evaluate your custom models with backtesting — see :doc:`backtesting`
- Explore the built-in transforms in the :doc:`../api/openstef_models/transforms` reference
- Review the presets for common configurations before building from scratch