Your First Forecast
===================

This tutorial walks you through building a complete energy forecast from scratch using OpenSTEF's custom pipeline approach. You'll learn how to prepare data, configure a workflow, train a model, generate probabilistic forecasts, and evaluate the results.

Unlike the :doc:`quickstart` which uses preset configurations, this guide gives you full control over each step so you understand what's happening under the hood.

.. mermaid:: /diagrams/getting_started/first_forecast_diagram_1.mmd

Overview
--------

A complete forecasting workflow in OpenSTEF involves five stages:

- **Data preparation** — Structure your time series into a ``TimeSeriesDataset``
- **Pipeline configuration** — Define transforms, model, and horizons in a ``CustomForecastingWorkflow``
- **Training** — Call ``fit()`` to preprocess data, train the model, and evaluate
- **Forecasting** — Call ``predict()`` to generate probabilistic predictions
- **Evaluation** — Inspect metrics and visualize results

Let's work through each step.

Step 1: Prepare Your Data
-------------------------

OpenSTEF expects input data as a ``TimeSeriesDataset`` — a pandas DataFrame with a ``DatetimeIndex`` at a consistent sampling interval. The dataset must include a target column (typically ``"load"``) and any weather features you want to use.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from openstef_core.datasets import TimeSeriesDataset

   # Create 3 months of hourly synthetic data
   n_samples = 24 * 31 * 3
   rng = np.random.default_rng(42)

   timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="h")

   # Simulate weather features
   temperature = 10 + 5 * np.sin(np.linspace(0, 6 * np.pi, n_samples)) + rng.standard_normal(n_samples)
   wind_speed = np.abs(5 + rng.standard_normal(n_samples) * 2)
   radiation = np.maximum(0, 200 * np.sin(np.linspace(0, 6 * np.pi, n_samples)) + rng.standard_normal(n_samples) * 20)

   # Simulate load: influenced by temperature, wind, and time-of-day patterns
   hour_pattern = 50 * np.sin(2 * np.pi * timestamps.hour / 24)
   load = 500 + 5 * temperature - 10 * wind_speed + hour_pattern + rng.standard_normal(n_samples) * 10

   data = pd.DataFrame(
       {
           "load": load,
           "temperature_2m": temperature,
           "wind_speed_10m": wind_speed,
           "shortwave_radiation": radiation,
       },
       index=timestamps,
   )

   dataset = TimeSeriesDataset(data=data)
   print(f"Dataset shape: {dataset.data.shape}")
   print(f"Time range: {dataset.data.index.min()} to {dataset.data.index.max()}")

The key requirement is that your DataFrame index is a regular ``DatetimeIndex`` with no gaps. OpenSTEF uses the temporal structure for feature engineering (lags, hour-of-day, etc.).

.. note::

   For real-world usage, you'd load data from your SCADA system, weather API, or database. OpenSTEF doesn't prescribe a data source — it works with any pandas DataFrame that meets the format above.

Step 2: Configure the Workflow
------------------------------

The ``CustomForecastingWorkflow`` ties together your model, preprocessing transforms, forecast horizons, and quantiles. Here we use the ``GBLinearForecaster`` — a gradient-boosted linear model well-suited for energy time series.

.. code-block:: python

   from openstef_core.types import LeadTime, Q
   from openstef_models.models.forecasting.gblinear_forecaster import (
       GBLinearForecaster,
   )
   from openstef_models.workflows import CustomForecastingWorkflow

   workflow = CustomForecastingWorkflow(
       model_id="my_first_forecast",
       forecasting_model=GBLinearForecaster(
           horizons=[LeadTime.from_string("PT36H")],  # Predict up to 36 hours ahead
           quantiles=[Q(0.5), Q(0.1), Q(0.9)],        # Median + 80% prediction interval
           target_column="load",
           temperature_column="temperature_2m",
           wind_speed_column="wind_speed_10m",
           radiation_column="shortwave_radiation",
           verbosity=1,
           gblinear_hyperparams=GBLinearForecaster.HyperParams(
               n_steps=50,  # Number of boosting iterations
           ),
       ),
   )
   print("Workflow configured successfully!")

**What each parameter means:**

- ``horizons`` — How far ahead to forecast. ``"PT36H"`` means 36 hours in ISO 8601 duration format.
- ``quantiles`` — Which prediction quantiles to generate. ``Q(0.5)`` is the median; ``Q(0.1)`` and ``Q(0.9)`` give an 80% prediction interval.
- ``target_column`` — The column name in your dataset that contains the value to predict.
- Weather columns — Tell the model which columns contain temperature, wind, radiation, etc., so it can apply appropriate feature engineering.

Step 3: Train the Model
-----------------------

Training is a single call to ``fit()``. Internally, this handles preprocessing (feature engineering, scaling, validation), model fitting on historical data, and evaluation on a held-out test split.

.. code-block:: python

   result = workflow.fit(dataset)

   if result is not None:
       print("Training complete!")
       print("\nFull Evaluation Metrics:")
       print(result.metrics_full.to_dataframe())

       if result.metrics_test is not None:
           print("\nTest Set Metrics (held-out validation):")
           print(result.metrics_test.to_dataframe())

The ``result`` object contains:

- ``metrics_full`` — Performance metrics computed on the full training set
- ``metrics_test`` — Metrics on a held-out validation split (more realistic estimate of future performance)

.. warning::

   Training requires sufficient historical data. For hourly data with a 36-hour horizon, aim for at least 4-6 weeks of history. Shorter datasets may produce unreliable models.

Step 4: Generate Forecasts
--------------------------

With the model trained, call ``predict()`` to generate probabilistic forecasts:

.. code-block:: python

   from openstef_core.datasets import ForecastDataset

   forecast: ForecastDataset = workflow.predict(dataset)

   # The forecast contains quantile predictions
   print(forecast.data.tail(10))

   # Access specific quantile series
   median = forecast.median_series        # P50 — best estimate
   quantiles = forecast.quantiles_data    # All quantile columns

The ``ForecastDataset`` provides:

- ``median_series`` — The P50 (median) forecast, your best single-point estimate
- ``quantiles_data`` — All requested quantiles as a DataFrame (P10, P50, P90)
- ``data`` — The full forecast DataFrame

.. note:: [VISUALIZATION: Time series plot showing historical load measurements as a solid line, the P50 median forecast as a dashed line, and the P10-P90 prediction interval as a shaded band]

Step 5: Evaluate and Visualize
------------------------------

Use the built-in ``ForecastTimeSeriesPlotter`` to create an interactive visualization comparing your forecast against measurements:

.. code-block:: python

   from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

   fig = (
       ForecastTimeSeriesPlotter()
       .add_measurements(measurements=dataset.data["load"])
       .add_model(
           model_name="gblinear",
           forecast=forecast.median_series,
           quantiles=forecast.quantiles_data,
       )
       .plot()
   )

   # Save as interactive HTML
   fig.write_html("my_first_forecast.html")

This produces an interactive Plotly chart showing measurements overlaid with your forecast and prediction intervals.

Understanding the Results
-------------------------

When evaluating your forecast, consider:

- **Calibration** — Does the 80% prediction interval (P10–P90) actually contain ~80% of observations? If not, the model is overconfident or underconfident.
- **Bias** — Is the median forecast systematically above or below actual values?
- **Sharpness** — Narrower prediction intervals are more useful, as long as they remain calibrated.

For systematic evaluation across multiple time periods, see the :doc:`backtesting` tutorial.

What's Next
-----------

Now that you understand the full pipeline:

- :doc:`quickstart` — See the same workflow with minimal boilerplate using presets
- :doc:`backtesting` — Evaluate your model on historical data with realistic train/test splits
- :doc:`advanced_customization` — Add custom transforms, swap models, tune hyperparameters, and integrate MLflow tracking