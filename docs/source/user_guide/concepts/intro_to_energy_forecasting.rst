Intro to Energy Forecasting
===========================

This page provides the conceptual foundation for understanding short-term energy forecasting — the core problem that OpenSTEF solves. If you're new to the domain, start here before diving into the library's architecture and APIs.

What Is Short-Term Energy Forecasting?
--------------------------------------

Short-term energy forecasting (STEF) is the task of predicting electrical load or generation at specific points in the grid over horizons ranging from 15 minutes to approximately 7 days. Grid operators rely on these forecasts to make operational decisions: managing congestion, planning transport capacity, coordinating with upstream and downstream network operators, and minimizing grid losses.

Unlike long-term planning (months to years), short-term forecasting must be:

- **Granular** — typically at 15-minute resolution
- **Frequent** — updated multiple times per day as new data arrives
- **Actionable** — accurate enough to trigger real operational decisions

.. mermaid:: /diagrams/user_guide/concepts/intro_to_energy_forecasting_diagram_1.mmd

Why Does It Matter?
-------------------

Grid operators face a fundamental challenge: electricity supply and demand must balance at every moment, yet the grid has physical capacity limits. Accurate forecasts enable:

- **Congestion management** — predicting when substations approach capacity limits so mitigation strategies can be deployed in time
- **Transport coordination** — communicating expected load to upstream transmission operators (e.g., a distribution operator reporting to a TSO)
- **Grid loss optimization** — minimizing financial costs by anticipating losses relative to market prices
- **Capacity planning** — understanding whether infrastructure can handle expected demand

The consequences of poor forecasts are tangible: unexpected congestion can damage equipment, force expensive emergency measures, or require curtailment of renewable generation.

Input Signals and Why They Matter
---------------------------------

Energy demand and generation are driven by a combination of physical, behavioural, and economic factors. Each input signal captures a different driver:

.. list-table:: Key Input Signals for Energy Forecasting
   :header-rows: 1
   :widths: 20 40 40

   * - Signal
     - Why It Matters
     - OpenSTEF Approach
   * - **Load history**
     - Energy demand is highly auto-correlated — last Monday's load is the best predictor of this Monday's load
     - :class:`~openstef_models.transforms.time_domain.LagsAdder` automatically selects valid lags respecting the forecast horizon
   * - **Weather forecasts**
     - Temperature drives heating/cooling demand; radiation and wind drive renewable generation
     - :class:`~openstef_models.transforms.time_domain.VersionedLagsAdder` respects data availability timestamps of weather forecasts
   * - **Calendar features**
     - Human behaviour follows daily, weekly, and holiday patterns (e.g., "if hour >= 17 and hour <= 20 → evening peak")
     - :class:`~openstef_models.transforms.time_domain.DatetimeFeaturesAdder`, :class:`~openstef_models.transforms.time_domain.HolidayFeatureAdder`, :class:`~openstef_models.transforms.time_domain.CyclicFeaturesAdder`
   * - **Market prices**
     - Energy prices influence behaviour — wind parks may shut down at negative prices; industrial users shift load
     - Configurable energy price columns in the forecasting workflow

.. note::

   Weather forecasts are *versioned*: the forecast for Monday issued on Saturday differs from the one issued on Sunday. Standard lag features ignore this distinction, which is why OpenSTEF provides specialized versioned lag handling.

Forecast Horizons and Accuracy Degradation
------------------------------------------

The **forecast horizon** (or lead time) is the time gap between when a prediction is made and the period it covers. OpenSTEF supports horizons from 15 minutes to approximately 7 days.

A critical constraint: **lags must respect the forecast horizon.** If you're forecasting 36 hours ahead, you cannot use data from 24 hours ago — it won't be available at prediction time. OpenSTEF's lag transforms enforce this automatically.

Accuracy degrades with increasing lead time for fundamental reasons:

- **Weather forecast quality drops** — beyond 7 days, weather models lack the 15-minute resolution needed for solar/wind peaks
- **Behavioural uncertainty compounds** — unpredictable events become more likely over longer windows
- **Auto-correlation weakens** — the further ahead you look, the less today's load tells you about future load

.. note:: [VISUALIZATION: Plot showing forecast error (e.g., rMAE) increasing as a function of lead time from 15 minutes to 7 days, with annotations showing the "sweet spot" for different use cases]

This is why OpenSTEF treats horizon as a first-class concept throughout its pipeline — from feature construction to model training to evaluation.

Challenges in the Field
------------------------

Energy forecasting is not a clean textbook regression problem. Practitioners face several domain-specific difficulties:

Unpredictable behaviour
^^^^^^^^^^^^^^^^^^^^^^^

Some energy users and generators behave in ways that are difficult to model from historical patterns alone:

- **Wind parks shutting down at negative market prices** — generation drops to zero not because of weather, but because of economic signals
- **Maintenance events** — planned or unplanned outages cause sudden load changes
- **Behind-the-meter solar and storage** — invisible generation that changes the apparent load profile
- **New connections or disconnections** — the underlying capacity of a grid point changes over time

Aggregation level effects
^^^^^^^^^^^^^^^^^^^^^^^^^

A fundamental principle: **higher aggregation levels are generally easier to forecast.** Individual customer behaviour is erratic, but the aggregate of thousands of customers follows smooth, predictable patterns. This means:

- Substation-level forecasts (many customers) are more predictable than individual customer forecasts
- System-level patterns (temporal, cyclic) dominate at high aggregation, while weather effects diminish
- Low-aggregation forecasts require different model optimization strategies (e.g., emphasis on peak detection rather than average accuracy)

OpenSTEF supports use cases across the full aggregation spectrum — from highly aggregated grid loss forecasting to individual customer predictions for congestion management.

Data quality as the binding constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In practice, **feature quality and data quality are the primary determinants of forecast accuracy** — often more so than model choice. Common issues include:

- Missing or delayed measurements from SCADA systems
- Weather forecast providers changing their model or grid resolution
- Incorrect meter configurations reporting wrong values
- Capacity changes not reflected in historical data

OpenSTEF provides transforms to extract maximum value from available data (rolling aggregates, cyclic encodings, holiday detection), but no amount of feature engineering can compensate for fundamentally unreliable inputs. Good input data is essential.

.. warning::

   "Garbage in, garbage out" applies strongly to energy forecasting. Before investing time in model tuning, verify that your input data is complete, correctly timestamped, and representative of current grid conditions.

How OpenSTEF Addresses These Challenges
----------------------------------------

OpenSTEF's design reflects lessons learned from years of operational forecasting at scale:

- **Horizon-aware feature engineering** — transforms automatically respect what data is available at each lead time
- **Modular transform pipelines** — compose feature extractors for your specific use case and data availability
- **Probabilistic forecasting** — quantile predictions capture uncertainty that increases with horizon
- **Extensible architecture** — add custom transforms or models without modifying core code

For a detailed walkthrough of how to build and run forecasts, see :ref:`guide_forecasting`. For information on available models, see :doc:`models`. For understanding how OpenSTEF handles uncertainty across model combinations, see :doc:`beam`.

Next Steps
----------

- :doc:`models` — understand which model types are available and when to use each
- :doc:`component_splitting` — learn how forecasts can be decomposed into solar, wind, and other components
- :ref:`guide_forecasting` — practical guide to building your first forecast pipeline
- :doc:`/tutorials/feature_engineering` — hands-on tutorial demonstrating feature transforms