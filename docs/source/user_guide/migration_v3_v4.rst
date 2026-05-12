Migrating from V3 to V4
=======================

This guide covers the breaking changes between OpenSTEF V3 and V4, and provides a
step-by-step workflow for migrating your existing code. V4 introduces a redesigned
architecture with separate packages, a workflow-based API, and Pydantic-based
configuration replacing the monolithic prediction job dictionary.

.. mermaid:: /diagrams/user_guide/migration_v3_v4_diagram_1.mmd

Overview of Breaking Changes
----------------------------

V4 is a ground-up redesign. The key changes are:

- **Package restructuring**: The single ``openstef`` package is split into ``openstef_core`` (datasets, base classes), ``openstef_models`` (forecasting models, workflows), and ``openstef_meta`` (preset workflows, ensemble orchestration).
- **PredictionJobDataClass removed**: Replaced by typed configuration classes using Pydantic ``BaseModel``/``BaseConfig``.
- **Pipeline functions replaced by Workflow classes**: ``train_model_pipeline()`` and ``create_forecast_pipeline()`` are replaced by ``workflow.fit()`` and ``workflow.predict()``.
- **DataFrames replaced by Dataset objects**: Raw pandas DataFrames are wrapped in ``TimeSeriesDataset`` and ``ForecastDataset`` with explicit metadata (sample interval, versioning).
- **MLflow integration redesigned**: Storage is now configured via ``MLFlowStorage`` objects passed to workflow configuration.
- **Callback system added**: Lifecycle hooks replace ad-hoc logging and validation.

Package Structure Changes
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - V3 Import
     - V4 Equivalent
   * - ``openstef.data_classes.prediction_job``
     - ``openstef_models.workflows`` (config classes)
   * - ``openstef.pipeline.train_model``
     - ``openstef_models.workflows``
   * - ``openstef.pipeline.create_forecast``
     - ``openstef_models.workflows``
   * - ``openstef.model_selection``
     - ``openstef_core.datasets``
   * - ``openstef.feature_engineering``
     - ``openstef_models.models.forecasting`` (preprocessing)

Install V4 packages:

.. code-block:: bash

   pip install openstef-core openstef-models openstef-meta

Step-by-Step Migration
----------------------

Step 1: Replace PredictionJobDataClass with Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before (V3):**

.. code-block:: python

   from openstef.data_classes.prediction_job import PredictionJobDataClass

   pj = dict(
       id=287,
       model='xgb',
       quantiles=[10, 30, 50, 70, 90],
       forecast_type="demand",
       lat=52.0,
       lon=5.0,
       horizon_minutes=47 * 60,
       resolution_minutes=15,
       name="Example",
       hyper_params={},
       feature_names=None,
       default_modelspecs=None,
       save_train_forecasts=True,
   )
   pj = PredictionJobDataClass(**pj)

**After (V4):**

.. code-block:: python

   from datetime import timedelta
   from openstef_meta.presets.forecasting_workflow import (
       EnsembleForecastingWorkflowConfig,
       create_ensemble_forecasting_workflow,
   )

   config = EnsembleForecastingWorkflowConfig(
       model_id="example_287",
       quantiles=[0.10, 0.30, 0.50, 0.70, 0.90],
       sample_interval=timedelta(minutes=15),
       forecast_horizon=timedelta(hours=47),
       mlflow_storage=MLFlowStorage(
           tracking_uri="./mlflow_tracking",
           local_artifacts_path="./mlflow_tracking_artifacts",
       ),
   )

**What changed:** The flat dictionary with mixed concerns (model config, storage paths, metadata) is replaced by a typed Pydantic configuration object. Quantiles are now expressed as floats between 0 and 1. Time parameters use ``timedelta`` objects instead of integer minutes.

Step 2: Replace Raw DataFrames with Dataset Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before (V3):**

.. code-block:: python

   import pandas as pd

   input_data = pd.read_csv(
       'data/get_model_input_pid_287.csv',
       index_col='index',
       parse_dates=True,
   )
   train_data = input_data.iloc[:-200, :]

**After (V4):**

.. code-block:: python

   import pandas as pd
   from datetime import timedelta
   from openstef_core.datasets import TimeSeriesDataset

   raw_data = pd.read_csv(
       'data/get_model_input_pid_287.csv',
       index_col='index',
       parse_dates=True,
   )

   dataset = TimeSeriesDataset(
       data=raw_data,
       sample_interval=timedelta(minutes=15),
   )

**What changed:** ``TimeSeriesDataset`` wraps the DataFrame with metadata about the time series (sample interval, versioning). This enables automatic validation and consistent handling across the pipeline.

Step 3: Replace Pipeline Functions with Workflow Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before (V3):**

.. code-block:: python

   from openstef.pipeline.train_model import train_model_pipeline
   from openstef.pipeline.create_forecast import create_forecast_pipeline

   # Train
   train, val, test = train_model_pipeline(
       pj,
       train_data,
       check_old_model_age=False,
       mlflow_tracking_uri="./mlflow_trained_models",
       artifact_folder="./mlflow_artifacts",
   )

   # Forecast
   forecast = create_forecast_pipeline(
       pj,
       forecast_data,
       model_folder="./mlflow_trained_models",
   )

**After (V4):**

.. code-block:: python

   from openstef_meta.presets.forecasting_workflow import (
       create_ensemble_forecasting_workflow,
   )

   workflow = create_ensemble_forecasting_workflow(config)

   # Train
   result = workflow.fit(dataset)
   if result is not None:
       print(result.metrics_full.to_dataframe())

   # Forecast
   forecast = workflow.predict(dataset)
   print(forecast.median_series)
   print(forecast.quantiles_data)

**What changed:** The stateless pipeline functions are replaced by a stateful workflow object. Training and prediction are methods on the same object, which manages model state, storage, and callbacks internally. The return types are rich objects (``ForecastDataset``) rather than raw DataFrames.

.. mermaid:: /diagrams/user_guide/migration_v3_v4_diagram_2.mmd

Step 4: Update Evaluation Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before (V3):**

.. code-block:: python

   # V3 evaluation was typically manual comparison of DataFrames
   import numpy as np

   mae = np.mean(np.abs(forecast['forecast'] - actual['load']))

**After (V4):**

.. code-block:: python

   from openstef_models.evaluation import EvaluationPipeline, EvaluationConfig

   eval_config = EvaluationConfig(
       lead_times=[LeadTime.from_string("PT36H")],
   )

   evaluation = EvaluationPipeline(
       config=eval_config,
       quantiles=config.quantiles,
       window_metric_providers=[...],
       global_metric_providers=[...],
   )

**What changed:** V4 provides a structured evaluation pipeline that computes metrics across multiple dimensions (lead times, availability windows, rolling periods) and always includes calibration metrics.

Step 5: Add Callbacks (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

V4 introduces a callback system for monitoring workflow lifecycle events:

.. code-block:: python

   from openstef_models.mixins.callbacks import PredictorCallback

   class MonitoringCallback(PredictorCallback):
       def on_fit_end(self, workflow, data):
           print("Training completed successfully")

       def on_predict_end(self, workflow, data, result):
           print(f"Forecast generated: {len(result.data)} rows")

   # Pass callbacks to workflow configuration
   workflow = create_ensemble_forecasting_workflow(config)

Common Migration Issues
-----------------------

**Quantile format change**
   V3 used integer percentiles (``[10, 30, 50, 70, 90]``). V4 uses float probabilities (``[0.10, 0.30, 0.50, 0.70, 0.90]``).

**No more ``check_old_model_age``**
   Model staleness checks are handled by the workflow's storage layer automatically.

**Feature engineering is internal**
   V3 exposed feature engineering as separate steps. In V4, preprocessing is configured as part of the ``ForecastingModel`` and applied automatically during ``fit()`` and ``predict()``.

**MLflow paths changed**
   V3 used ``mlflow_tracking_uri`` and ``artifact_folder`` as string parameters. V4 uses a structured ``MLFlowStorage`` configuration object.

.. warning::

   V3 and V4 model artifacts are **not compatible**. You must retrain all models after migrating to V4. There is no automatic model conversion.

Migration Checklist
-------------------

- Replace ``pip install openstef`` with ``pip install openstef-core openstef-models openstef-meta``
- Replace ``PredictionJobDataClass`` with appropriate V4 config classes
- Wrap input DataFrames in ``TimeSeriesDataset``
- Replace ``train_model_pipeline()`` with ``workflow.fit()``
- Replace ``create_forecast_pipeline()`` with ``workflow.predict()``
- Update quantile format from integers to floats
- Update MLflow configuration to use ``MLFlowStorage``
- Retrain all models (V3 artifacts are incompatible)
- Replace manual evaluation with ``EvaluationPipeline``

For production deployment patterns after migration, see :doc:`deployment`. For data
integration with external sources, see :doc:`data_integration`.