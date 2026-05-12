Quickstart
==========

Get from zero to a working forecast in under a minute. This page provides a
copy-paste-ready minimal example using OpenSTEF's workflow presets. No
explanations—just working code. For the *why* behind each step, see
:doc:`first_forecast`.

.. mermaid:: /diagrams/getting_started/quickstart_diagram_1.mmd

Prerequisites
-------------

Make sure OpenSTEF is installed. If not, see :doc:`installation`.

.. code-block:: bash

   pip install openstef-models

Minimal Working Example
-----------------------

The following script creates synthetic energy load data, configures a forecasting
workflow, trains a model, and produces a forecast—all in one go.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from datetime import timedelta

   from openstef_models.presets.forecasting_workflow import (
       ForecastingWorkflowConfig,
       create_forecasting_workflow,
   )
   from openstef_models.data import TimeSeriesDataset

   # --- Step 1: Create synthetic data ---
   # 30 days of 15-minute resolution load data with a daily pattern
   periods = 30 * 24 * 4  # 30 days at 15-min intervals
   index = pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC")

   np.random.seed(42)
   hour = index.hour + index.minute / 60.0
   load = 50 + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2, periods)

   df = pd.DataFrame({"load": load}, index=index)
   df.index.name = "datetime"

   # --- Step 2: Configure the workflow ---
   config = ForecastingWorkflowConfig(
       target_column="load",
       horizons=[timedelta(hours=24)],
       selected_features=["load"],
       model_type="xgb_quantile",
       quantiles=(0.1, 0.5, 0.9),
   )

   # --- Step 3: Create the workflow ---
   workflow = create_forecasting_workflow(config)

   # --- Step 4: Split data into train / predict sets ---
   train_df = df.iloc[:-96]   # all but last day
   predict_df = df.iloc[-96:]  # last day (96 quarter-hours)

   train_data = TimeSeriesDataset(train_df)
   predict_data = TimeSeriesDataset(predict_df)

   # --- Step 5: Fit the model ---
   fit_result = workflow.fit(train_data)

   # --- Step 6: Generate a forecast ---
   forecast = workflow.predict(predict_data)

   # --- Step 7: Inspect results ---
   print(forecast.to_dataframe().head())

What Just Happened
------------------

In six lines of meaningful code you:

- Created a ``ForecastingWorkflowConfig`` that defines the target column, forecast
  horizon, features, model type, and output quantiles.
- Called ``create_forecasting_workflow`` which assembled the full pipeline—preprocessing,
  feature engineering, model, and postprocessing—from that single config object.
- Trained the model with ``workflow.fit()``.
- Produced probabilistic forecasts with ``workflow.predict()``.

The output is a ``TimeSeriesDataset`` containing columns for each requested quantile
(p10, p50, p90), giving you both a point forecast and confidence intervals.

Key Configuration Options
-------------------------

The ``ForecastingWorkflowConfig`` accepts many parameters. Here are the ones you'll
adjust most often:

.. code-block:: python

   config = ForecastingWorkflowConfig(
       target_column="load",                    # column name to forecast
       horizons=[timedelta(hours=24)],          # forecast horizons
       selected_features=["load", "temperature", "wind_speed"],  # input features
       model_type="xgb_quantile",              # model backend
       quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),  # output quantiles
       completeness_threshold=0.7,             # minimum data completeness
       flatliner_threshold=24,                 # hours before flagging flatline
       verbosity=1,                            # 0=silent, 1=warning, 2=info, 3=debug
   )

Next Steps
----------

- :doc:`first_forecast` — Detailed walkthrough explaining each concept
- :doc:`backtesting` — Evaluate model performance on historical data
- :doc:`advanced_customization` — Swap models, add custom transforms, tune hyperparameters