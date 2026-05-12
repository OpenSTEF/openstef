Feature Engineering for Energy Forecasting
==========================================

Feature engineering is the process of transforming raw input data into informative predictors that improve forecasting accuracy. In energy forecasting, domain-specific features—derived from weather, time patterns, and historical load—are often more impactful than model complexity. OpenSTEF provides a suite of built-in feature transforms that encode energy domain knowledge directly into the forecasting pipeline.

This page covers the types of features used in OpenSTEF, how they are constructed, and how to configure them for your forecasting workflows.


Why Features Matter for Energy Forecasting
-------------------------------------------

Classical ML models like XGBoost and LightGBM—the workhorses of OpenSTEF—achieve high performance through smart feature engineering rather than architectural complexity. The key insight is that energy consumption and generation follow predictable patterns driven by:

- **Weather conditions** — temperature, solar radiation, wind speed, and humidity directly influence both demand and renewable generation.
- **Time of day and week** — human activity follows cyclical patterns that repeat daily and weekly.
- **Historical load** — yesterday's peak predicts today's peak; the previous hour's demand influences the next hour's.
- **Calendar effects** — holidays, weekends, and seasons shift consumption patterns.

OpenSTEF encodes this domain knowledge through a pipeline of *feature adders*—transforms that augment raw time series data with derived columns before model training.

.. mermaid:: /diagrams/concepts/feature_engineering_diagram_1.mmd

Built-in Feature Transforms
----------------------------

When you create a forecasting workflow via ``create_forecasting_workflow``, OpenSTEF automatically assembles a feature engineering pipeline. The core feature adders are:

Lag Features (LagsAdder)
^^^^^^^^^^^^^^^^^^^^^^^^^

Lag features capture temporal dependencies by creating shifted copies of the target variable. OpenSTEF's ``LagsAdder`` is aware of forecast horizons—it only creates lags using data that would have been available at prediction time.

.. code-block:: python

   from datetime import timedelta
   from openstef_models.transforms.time_domain.lags_adder import LagsAdder

   lags_adder = LagsAdder(
       history_available=timedelta(hours=48),
       horizons=[timedelta(hours=1), timedelta(hours=24)],
       add_trivial_lags=True,
       target_column="load",
       max_day_lags=14,
       custom_lags=[timedelta(days=7)],
   )

The ``LagsAdder`` supports multiple strategies:

- **Trivial lags** — minute-based lags for short-term patterns (e.g., 1h, 2h ago) and day-based lags for daily/weekly patterns (e.g., 1 day, 7 days ago).
- **Custom lags** — explicitly specified lag durations, such as a 7-day lag for weekly seasonality.
- **Autocorrelation-based lags** — adaptive selection based on the signal's autocorrelation structure.

For linear models like ``gblinear``, OpenSTEF uses only a 7-day lag to avoid multicollinearity, while tree-based models benefit from richer lag sets.

Wind Power Features (WindPowerFeatureAdder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts raw wind speed into estimated wind power using the cubic relationship between wind speed and power output:

.. code-block:: python

   from openstef_models.transforms.time_domain import WindPowerFeatureAdder

   wind_adder = WindPowerFeatureAdder(
       windspeed_reference_column="windspeed_100m",
   )

Atmosphere-Derived Features (AtmosphereDerivedFeaturesAdder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combines pressure, humidity, and temperature into derived quantities that better predict energy demand (e.g., apparent temperature, dew point):

.. code-block:: python

   from openstef_models.transforms.time_domain import AtmosphereDerivedFeaturesAdder

   atmo_adder = AtmosphereDerivedFeaturesAdder(
       pressure_column="pressure",
       relative_humidity_column="humidity",
       temperature_column="temperature",
   )

Radiation-Derived Features (RadiationDerivedFeaturesAdder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transforms raw radiation data into features relevant for solar generation forecasting, accounting for the sun's position at a given geographic coordinate:

.. code-block:: python

   from openstef_core.types import Coordinate, Latitude, Longitude
   from decimal import Decimal
   from openstef_models.transforms.time_domain import RadiationDerivedFeaturesAdder

   radiation_adder = RadiationDerivedFeaturesAdder(
       coordinate=Coordinate(
           latitude=Latitude(Decimal("52.13")),
           longitude=Longitude(Decimal("5.29")),
       ),
       radiation_column="radiation",
   )

Cyclic Time Features (CyclicFeaturesAdder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Encodes time-of-day and day-of-week as sine/cosine pairs, preserving the cyclical nature (11 PM is close to midnight, Sunday is close to Monday):

.. code-block:: python

   from openstef_models.transforms.time_domain import CyclicFeaturesAdder

   cyclic_adder = CyclicFeaturesAdder()

This avoids the discontinuity problem of raw hour/weekday integers and allows models to learn smooth periodic patterns.

Daylight Features (DaylightFeatureAdder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds sunrise/sunset-derived features for a given location, capturing the seasonal variation in daylight hours that drives both lighting demand and solar generation:

.. code-block:: python

   from openstef_models.transforms.time_domain import DaylightFeatureAdder

   daylight_adder = DaylightFeatureAdder(
       coordinate=Coordinate(
           latitude=Latitude(Decimal("52.13")),
           longitude=Longitude(Decimal("5.29")),
       ),
   )

Rolling Aggregate Features (RollingAggregatesAdder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computes rolling statistics (mean, std, min, max) over the target variable to capture trends and volatility:

.. code-block:: python

   from openstef_models.transforms.time_domain import RollingAggregatesAdder

   rolling_adder = RollingAggregatesAdder(
       feature="load",
       aggregation_functions=["mean", "std"],
       horizons=[timedelta(hours=1), timedelta(hours=24)],
   )


Configuring Features via ForecastingWorkflowConfig
---------------------------------------------------

Rather than assembling feature adders manually, you typically configure them through ``ForecastingWorkflowConfig``. The configuration controls which columns map to which weather variables, the geographic location (for solar/daylight calculations), and lag behavior:

.. code-block:: python

   from openstef_models.workflows.config import ForecastingWorkflowConfig

   config = ForecastingWorkflowConfig(
       model_id="my_substation_01",
       model="xgboost",
       target_column="load",
       wind_speed_column="windspeed_100m",
       temperature_column="temperature",
       pressure_column="pressure",
       relative_humidity_column="humidity",
       radiation_column="radiation",
       location={"coordinate": {"latitude": "52.13", "longitude": "5.29"}},
       max_day_lags=14,
       rolling_aggregate_features=["mean", "std"],
   )

The ``create_forecasting_workflow`` function reads this configuration and builds the complete feature pipeline automatically.


What Makes a Good Feature for Energy Forecasting
--------------------------------------------------

When designing custom features or selecting input data, consider these principles:

- **Causal relevance** — The feature should have a physical or behavioral relationship to energy consumption or generation. Temperature affects heating/cooling load; wind speed drives turbine output.
- **Availability at prediction time** — A feature is useless if it won't be available when you need to make a forecast. Weather forecasts degrade beyond 7 days; real-time measurements have reporting delays.
- **Appropriate granularity** — Match the feature's temporal resolution to your forecast resolution. A daily average temperature is insufficient for 15-minute peak detection.
- **Non-redundancy** — Highly correlated features add noise without information. OpenSTEF's model-specific lag strategies (fewer lags for linear models) reflect this principle.
- **Stationarity** — Features that drift over time (e.g., raw timestamps) perform poorly. Cyclic encoding and differencing address this.

.. warning::

   Never include features derived from future values of the target variable. OpenSTEF's ``LagsAdder`` enforces this constraint automatically, but custom features must respect forecast horizon boundaries.


Feature Importance and Selection
---------------------------------

After training, you can inspect which features the model relies on most. OpenSTEF models that implement the ``ExplainableForecaster`` interface provide feature importance rankings:

.. code-block:: python

   from typing import cast
   from openstef_models.explainability import ExplainableForecaster

   explainable_model = cast(ExplainableForecaster, workflow.model.forecaster)
   fig = explainable_model.plot_feature_importances()
   fig.show()

.. note:: [VISUALIZATION: Feature importance treemap showing relative contribution of weather, lag, and time features to model predictions]

You can also control which features are used via the ``selected_features`` field in your configuration, which applies a ``Selector`` transform to filter the feature matrix before training.


Related Topics
--------------

- For an introduction to forecasting concepts, see :doc:`forecasting_basics`.
- For understanding model output as probability distributions, see :doc:`quantiles_and_confidence`.
- For how OpenSTEF combines multiple models, see :doc:`meta_ensembles`.
- For decomposing load into generation and consumption, see :doc:`component_splitting`.