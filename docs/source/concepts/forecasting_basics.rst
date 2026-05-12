Short-Term Forecasting Basics
=============================

This page explains the fundamentals of short-term energy forecasting: what it is, why grid operators and energy systems need it, and how it differs from long-term planning forecasts. You'll learn about forecast horizons, lead times, update frequency, and how these concepts map to OpenSTEF's configuration.

What Is Short-Term Energy Forecasting?
--------------------------------------

Short-term energy forecasting predicts electricity load (consumption or generation) over a horizon of **hours to approximately 7 days**. This contrasts with long-term forecasting, which projects energy demand over months or years for infrastructure planning.

In the short-term domain, forecasts must be:

- **High resolution** — typically 15-minute intervals matching grid measurement systems
- **Frequently updated** — new forecasts every few hours as fresh weather data arrives
- **Probabilistic** — expressing uncertainty through quantiles, not just point predictions (see :doc:`quantiles_and_confidence`)

Short-term forecasts degrade significantly beyond 7 days because weather forecasts lose 15-minute resolution, and solar/wind peaks become unpredictable at that range.

Why Short-Term Forecasting Matters
----------------------------------

Grid operators rely on short-term forecasts for operational decisions that cannot wait for human analysis:

- **Congestion management** — predicting peak loads at substations to trigger mitigation strategies before overloads occur
- **Transport forecasts** — communicating planned energy flows to upstream transmission operators (e.g., a distribution operator providing forecasts to the TSO)
- **Grid losses forecasting** — minimizing financial costs by anticipating system-level losses relative to market prices
- **Renewable integration** — anticipating solar and wind variability to balance supply and demand

Each use case emphasizes different aspects of forecast quality. Congestion management cares primarily about accuracy near peaks, while transport forecasts need balanced performance across the entire horizon.

Forecast Horizons
-----------------

The **forecast horizon** is how far into the future a prediction extends. OpenSTEF supports configurable horizons, commonly ranging from 1 hour to 48 hours (and up to 7 days in some configurations).

.. mermaid:: /diagrams/concepts/forecasting_basics_diagram_1.mmd

In OpenSTEF, horizons are expressed using the ``LeadTime`` type:

.. code-block:: python

   from openstef_core.types import LeadTime
   from datetime import timedelta

   # Common forecast horizons
   horizon_1h = LeadTime(timedelta(hours=1))
   horizon_36h = LeadTime(timedelta(hours=36))
   horizon_7d = LeadTime(timedelta(days=7))

   # Parse from ISO 8601 duration string
   horizon = LeadTime.from_string("PT36H")  # 36 hours ahead

Choosing the right horizon depends on your operational need:

- **15 min – 1 hour**: Real-time balancing, immediate grid actions
- **1 – 6 hours**: Intra-day market trading, short-term congestion response
- **6 – 36 hours**: Day-ahead planning, the most common operational window
- **36 hours – 7 days**: Extended planning, capacity scheduling

Lead Time
---------

**Lead time** is the duration between when a forecast is *created* and the moment being predicted. A forecast generated at 08:00 predicting load at 14:00 has a lead time of 6 hours.

Forecast accuracy degrades with increasing lead time. This is a fundamental property of all forecasting systems — not a limitation of any particular model. OpenSTEF's backtesting framework explicitly evaluates this degradation:

.. code-block:: python

   from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
   from openstef_core.types import LeadTime, Q

   workflow = create_forecasting_workflow(
       config=ForecastingWorkflowConfig(
           model_id="substation_forecast_v1",
           model="gblinear",
           horizons=[LeadTime.from_string("PT36H")],
           quantiles=[Q(0.5), Q(0.1), Q(0.9)],
           target_column="load",
       )
   )

The ``horizons`` parameter defines the maximum prediction window. The model produces forecasts at 15-minute sample intervals from the current time up to the specified horizon.

Update Frequency
----------------

**Update frequency** (or predict interval) determines how often new forecasts are generated. More frequent updates incorporate the latest weather data and measurements, improving accuracy for near-term predictions.

Typical configurations:

- **Every 15 minutes**: Maximum freshness, highest computational cost
- **Every 6 hours**: Standard operational cadence, balances accuracy and resources
- **Every 24 hours**: Day-ahead only, minimal compute

In OpenSTEF's backtesting configuration, this is controlled by ``predict_interval``:

.. code-block:: python

   from datetime import timedelta

   from openstef_beam.backtesting import BacktestPipelineConfig

   config = BacktestPipelineConfig(
       prediction_sample_interval=timedelta(minutes=15),  # Output resolution
       predict_interval=timedelta(hours=6),               # Generate new forecast every 6h
       train_interval=timedelta(days=7),                  # Retrain model weekly
   )

The interplay between these three intervals defines the operational rhythm:

- ``prediction_sample_interval`` — the time step between consecutive forecast points (typically 15 minutes)
- ``predict_interval`` — how often the system produces a fresh forecast
- ``train_interval`` — how often the model is retrained on new data

Short-Term vs. Long-Term Forecasting
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Characteristic
     - Short-Term (OpenSTEF)
     - Long-Term
   * - Horizon
     - Hours to 7 days
     - Months to decades
   * - Resolution
     - 15-minute intervals
     - Hourly, daily, or monthly
   * - Primary drivers
     - Weather, time-of-day, day-of-week
     - Economic growth, policy, demographics
   * - Update frequency
     - Every 15 min to 6 hours
     - Monthly or quarterly
   * - Key challenge
     - Weather uncertainty, sudden changes
     - Structural shifts, technology adoption
   * - Output type
     - Probabilistic (quantiles)
     - Scenario-based projections
   * - Typical models
     - Gradient boosting, neural networks
     - Regression, econometric models

Short-term forecasting is fundamentally a **weather-driven** problem. Temperature, solar radiation, and wind speed dominate prediction accuracy for most grid points. Long-term forecasting is an **economics-driven** problem where demographic and policy changes matter more than tomorrow's cloud cover.

For details on how weather and other predictors are used, see :doc:`feature_engineering`.

Forecast Quality and Degradation
--------------------------------

Forecast accuracy is not uniform across the prediction horizon. A useful mental model:

- **0–6 hours ahead**: High accuracy. Recent measurements and current weather provide strong signal.
- **6–24 hours ahead**: Good accuracy. Day-ahead weather forecasts are reliable for temperature and broad patterns.
- **24–48 hours ahead**: Moderate accuracy. Weather uncertainty grows, especially for solar radiation and wind.
- **48 hours – 7 days**: Decreasing accuracy. Weather forecasts lose fine-grained resolution; the model increasingly relies on seasonal and weekly patterns.

OpenSTEF provides tools to evaluate this degradation through lead time analysis in the backtesting framework. This helps you determine whether your operational decisions can rely on forecasts at a given lead time.

.. warning::

   Forecast quality depends heavily on input data quality and the aggregation level of the prediction target. Individual customer forecasts are far less predictable than substation-level aggregates due to behavioural variability.

Practical Implications for OpenSTEF Users
-----------------------------------------

When configuring an OpenSTEF forecasting workflow, consider:

1. **Match horizon to decision timing** — if your operational decision happens 12 hours before the event, a 36-hour horizon provides comfortable margin.

2. **Update as often as practical** — each update incorporates fresher weather data. A 6-hour predict interval is a sensible default.

3. **Use probabilistic outputs** — point forecasts hide uncertainty. Configure multiple quantiles to understand the range of possible outcomes (see :doc:`quantiles_and_confidence`).

4. **Evaluate by lead time** — don't average accuracy across the whole horizon. A model may excel at 1-hour predictions but struggle at 36 hours. Use OpenSTEF's backtesting to diagnose where performance breaks down.

5. **Consider fallback strategies** — when models fail or data is unavailable, a base-case forecaster using last week's pattern provides a reasonable fallback (see :doc:`reliability_and_fallback`).

Next Steps
----------

- :doc:`feature_engineering` — understand which weather and temporal features drive forecast accuracy
- :doc:`quantiles_and_confidence` — learn how probabilistic forecasts express uncertainty
- :doc:`reliability_and_fallback` — handle production failures gracefully