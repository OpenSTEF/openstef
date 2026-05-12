Common Use Cases
================

This page describes the common forecasting use cases supported by OpenSTEF, explains what makes each one different, and provides practical configuration examples. OpenSTEF's flexible ``ForecastingWorkflowConfig`` allows you to tailor model behaviour, metrics, and optimization targets to each scenario.

.. mermaid:: /diagrams/user_guide/use_cases_diagram_1.mmd

Overview
--------

OpenSTEF originated as a congestion management tool for Dutch grid operators, but has grown to support a range of energy forecasting scenarios. Each use case differs in:

- **Optimization target** — peak accuracy vs. overall accuracy vs. cost-weighted error
- **Aggregation level** — individual customer to system-wide
- **Key predictors** — weather-dominated vs. temporal-pattern-dominated
- **Output format** — point forecasts, quantiles, or component splits

The sections below describe each use case, when to choose it, and how to configure OpenSTEF accordingly.

Congestion Management Forecasts
-------------------------------

**When to use:** You need to predict peak load at substations, transformers, or cables to trigger mitigation actions before overload occurs.

**What makes it different:**

- Accuracy matters most near peak load periods, not across the full distribution
- Aggregation levels vary widely — from entire substations down to individual customers
- High-quantile predictions (e.g., P90, P95) are critical for risk assessment
- Key metrics: effective precision/recall at peaks, rMAE at 50th quantile during peaks, rCRPS

**Configuration example:**

.. code-block:: python

   from datetime import timedelta
   from openstef.configuration import ForecastingWorkflowConfig, Quantile

   congestion_config = ForecastingWorkflowConfig(
       model_id="substation_transformer_01",
       model="xgboost",
       quantiles=[
           Quantile(0.10),
           Quantile(0.50),
           Quantile(0.90),
           Quantile(0.95),
       ],
       sample_interval=timedelta(minutes=15),
       horizons=[LeadTime.from_string("PT24H"), LeadTime.from_string("PT48H")],
   )

**Tip:** For congestion management, configure sample weights to emphasize peak periods. XGBoost with exponential weighting works well for capturing tail behaviour at low-aggregation points.

Free Space Estimation
---------------------

**When to use:** You need to estimate remaining capacity on a transformer or cable — i.e., how much additional load can be connected before reaching thermal limits.

**What makes it different:**

- Free space = rated capacity − forecasted peak load
- Requires high-quantile forecasts (the upper bound of expected load)
- Often combined with congestion forecasts as a downstream calculation
- Output is expressed in MW or MVA of available headroom

**Approach:**

Free space estimation is not a separate model type — it uses the same congestion forecast pipeline but focuses on the upper quantiles. You subtract the high-quantile forecast from the asset's rated capacity:

.. code-block:: python

   rated_capacity_mw = 40.0  # Transformer rating

   # After running a forecast with high quantiles:
   free_space = rated_capacity_mw - forecast_results["quantile_0.95"]

This gives operators a conservative estimate of how much new load (e.g., EV chargers, heat pumps) can be connected without triggering congestion.

Transport Forecasts
-------------------

**When to use:** You need to communicate planned energy flows to upstream or downstream network operators (e.g., a DSO reporting to a TSO).

**What makes it different:**

- Overall accuracy across all time periods matters equally — not just peaks
- Medium aggregation levels provide a balance between predictability and granularity
- Some operators require component splits (solar, wind, other) as separate forecast streams
- Key metric: rMAE across the full forecast horizon
- Balanced performance is more important than tail accuracy

**Configuration example:**

.. code-block:: python

   transport_config = ForecastingWorkflowConfig(
       model_id="transport_forecast_region_north",
       model="lgbm",
       quantiles=[Quantile(0.50)],  # Point forecast sufficient
       sample_interval=timedelta(minutes=15),
       horizons=[LeadTime.from_string("PT24H"), LeadTime.from_string("PT48H")],
   )

For split-component transport forecasts, run separate models for solar, wind, and residual load, then combine the outputs.

Grid Loss Forecasts
-------------------

**When to use:** You need to predict system-level grid losses for financial optimization against market prices.

**What makes it different:**

- Highly aggregated — system-wide patterns dominate
- Weather predictors have diminished impact; temporal and cyclic patterns are stronger
- Error weighting based on market prices (a 1 MW error during a price spike costs more than during off-peak)
- Key metrics: rMAE plus total error cost minimization

**Configuration example:**

.. code-block:: python

   grid_loss_config = ForecastingWorkflowConfig(
       model_id="grid_losses_national",
       model="gblinear",  # Linear model suits smooth, aggregated patterns
       quantiles=[Quantile(0.50)],
       sample_interval=timedelta(minutes=15),
       horizons=[LeadTime.from_string("PT24H"), LeadTime.from_string("PT48H")],
       energy_price_column="EPEX_NL",  # Include market price as feature
   )

**Tip:** For grid losses, ``gblinear`` or ``lgbmlinear`` often outperform tree-based models because the underlying signal is smoother and more linear at high aggregation levels.

District Heating Demand
-----------------------

**When to use:** You need to forecast thermal energy demand for district heating networks — a non-electricity use case that demonstrates OpenSTEF's domain flexibility.

**What makes it different:**

- Target variable is thermal demand (MWth) rather than electrical load
- Temperature is the dominant predictor (stronger correlation than in electricity forecasting)
- Seasonal patterns are more pronounced
- Building thermal inertia introduces lag effects

**Configuration example:**

.. code-block:: python

   heating_config = ForecastingWorkflowConfig(
       model_id="district_heating_zone_a",
       model="xgboost",
       quantiles=[Quantile(0.10), Quantile(0.50), Quantile(0.90)],
       sample_interval=timedelta(hours=1),  # Hourly resolution typical
       horizons=[LeadTime.from_string("PT48H")],
       temperature_column="temperature_2m",
   )

OpenSTEF's feature engineering (lag features, rolling aggregates) captures thermal inertia effects well without custom preprocessing.

MV Route Congestion with Power-Grid-Model
------------------------------------------

**When to use:** You need topology-aware forecasts that account for how power flows through medium-voltage routes, not just individual substations.

**What makes it different:**

- Combines OpenSTEF's time-series forecasting with `power-grid-model <https://github.com/PowerGridModel/power-grid-model>`_ network topology calculations
- Forecasts at individual nodes are propagated through the network model to identify route-level congestion
- Captures the effect of distributed generation on line loading
- Requires network topology data in addition to measurement time series

**Approach:**

1. Train individual forecasting models for each measurement point on the MV route
2. Run forecasts to produce per-node load/generation predictions
3. Feed predictions into power-grid-model as injection values
4. Run power flow calculations to determine line and cable loading

.. code-block:: python

   # Step 1-2: Generate forecasts for each node
   node_forecasts = {}
   for node_config in mv_route_node_configs:
       workflow = ForecastingWorkflow(node_config)
       node_forecasts[node_config.model_id] = workflow.predict(input_data)

   # Step 3-4: Use power-grid-model for topology-aware loading
   # (power-grid-model is a separate library — see its documentation)
   import power_grid_model as pgm

   model = pgm.PowerGridModel(input_data=network_topology)
   # Inject forecasted values and run power flow...

This combined approach reveals congestion that single-point forecasts would miss — for example, when distributed solar on one feeder causes reverse power flow that overloads a cable segment.

Choosing the Right Use Case
---------------------------

+---------------------------+-------------------+------------------+----------------------------+
| Use Case                  | Aggregation       | Key Metric       | Recommended Model          |
+===========================+===================+==================+============================+
| Congestion management     | Low to high       | Peak accuracy    | ``xgboost``                |
+---------------------------+-------------------+------------------+----------------------------+
| Free space estimation     | Low to medium     | Upper quantile   | ``xgboost``                |
+---------------------------+-------------------+------------------+----------------------------+
| Transport forecasts       | Medium            | rMAE             | ``lgbm``                   |
+---------------------------+-------------------+------------------+----------------------------+
| Grid losses               | Very high         | Cost-weighted    | ``gblinear``/``lgbmlinear``|
+---------------------------+-------------------+------------------+----------------------------+
| District heating          | Medium            | rMAE             | ``xgboost``                |
+---------------------------+-------------------+------------------+----------------------------+
| MV route congestion       | Per-node + topo   | Line loading     | ``xgboost`` + pgm          |
+---------------------------+-------------------+------------------+----------------------------+

Related Pages
-------------

- See :doc:`data_integration` for how to connect measurement data sources to these workflows
- See :doc:`deployment` for running these use cases in production environments
- See :doc:`migration_v3_v4` if you are upgrading existing use case configurations from V3