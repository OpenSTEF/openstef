Component Splitting
===================

.. _concept_component_splitting:

This page explains **component splitting** — the process of decomposing an aggregated energy forecast into its constituent parts (solar, wind, base load, etc.). This is a post-forecast step that helps grid operators understand *what* is driving the load they see at a substation or grid connection point.

Why Component Splitting Matters
-------------------------------

A grid operator typically observes a single aggregated load signal at a measurement point — the net sum of all generation and consumption behind that connection. But for operational decision-making, knowing the *composition* of that load is critical:

- **Congestion analysis**: Understanding how much solar generation is expected helps predict when reverse power flows might cause congestion.
- **Grid planning**: Knowing the wind vs. solar vs. base-load mix informs infrastructure investment decisions.
- **Regulatory reporting**: Transmission system operators often require breakdowns by generation type.
- **Curtailment decisions**: If congestion is forecast, operators need to know which generation source to curtail.

.. warning::

   Component splitting does **not** improve forecast accuracy. The forecast is produced first on the aggregated load signal (see :doc:`intro_to_energy_forecasting`), and *then* decomposed into components. The total of all components always equals the original forecast.

.. mermaid:: /diagrams/user_guide/concepts/component_splitting_diagram_1.mmd

How It Works
------------

The core invariant of component splitting is simple: **the predicted components must sum to the original source values**. The splitter takes a time series of total load and produces a time series for each component, using either:

- **Known ratios** — fixed proportions (e.g., a connection point is 60% solar, 40% wind)
- **Weather-correlated models** — a pre-trained linear model that uses radiation and wind speed to estimate how much of the total load is attributable to each source

OpenSTEF provides an abstract interface (:class:`~openstef_models.models.component_splitting.ComponentSplitter`) and concrete implementations for both approaches.

Available Splitters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Splitter
     - Approach
     - When to Use
   * - :class:`~openstef_models.models.component_splitting.constant_component_splitter.ConstantComponentSplitter`
     - Applies fixed ratios per component type
     - Known solar parks, wind farms, or fixed generation mixes
   * - :class:`~openstef_models.models.component_splitting.linear_component_splitter.LinearComponentSplitter`
     - Pre-trained linear model using radiation and wind speed features
     - Mixed connection points where component shares vary with weather

Constant Component Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest approach: you know the generation mix and apply fixed ratios. This is appropriate for dedicated generation assets (a pure solar park, a wind farm) or connection points with well-characterized installed capacity.

.. code-block:: python

   from openstef_models.models.component_splitting.constant_component_splitter import (
       ConstantComponentSplitter, ConstantComponentSplitterConfig
   )
   from openstef_core.types import EnergyComponentType

   config = ConstantComponentSplitterConfig(
       source_column="total_load",
       component_ratios={EnergyComponentType.SOLAR: 0.6, EnergyComponentType.WIND: 0.4},
   )
   splitter = ConstantComponentSplitter(config)

Convenience factory methods are also available:

.. code-block:: python

   solar_splitter = ConstantComponentSplitter.known_solar_park()
   wind_splitter = ConstantComponentSplitter.known_wind_farm()

Linear Component Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^^

For mixed connection points, the :class:`~openstef_models.models.component_splitting.linear_component_splitter.LinearComponentSplitter` uses weather features (radiation and wind speed at 100m) to estimate the contribution of each component. It decomposes total load into three predefined components:

- **Wind on shore**
- **Solar**
- **Other** (residual: base industrial, residential, etc.)

This splitter uses a pre-trained model and does not currently support re-training. It requires ``radiation`` and ``windspeed_100m`` columns in the input data alongside the source load column.

The Component Splitting Pipeline
---------------------------------

For production use, OpenSTEF provides :class:`~openstef_models.models.component_splitting_model.ComponentSplittingModel` — a high-level orchestrator that combines preprocessing, splitting, and postprocessing into a single pipeline:

.. code-block:: python

   from openstef_models.models.component_splitting_model import ComponentSplittingModel

   model = ComponentSplittingModel(
       component_splitter=splitter,
       preprocessing=preprocessing_pipeline,
       source_column="total_load",
   )
   model.fit(training_data)
   components = model.predict(new_data)

The ``predict()`` method returns an :class:`~openstef_core.types.EnergyComponentDataset` — a structured dataset where each component is a separate column, and the sum across components equals the original source column.

.. note:: [VISUALIZATION: Stacked area chart showing a 48-hour forecast decomposed into solar (yellow), wind (blue), and other/residual (grey) components, with the total load line overlaid on top demonstrating that components sum to the forecast.]

Key Design Decisions
--------------------

**Forecast first, split second**
   OpenSTEF deliberately separates forecasting from decomposition. Forecasting on the aggregated signal benefits from the smoothing effect of aggregation and avoids error accumulation from forecasting each component independently. The split is a downstream interpretation step.

**Sum-preserving constraint**
   All splitters enforce the invariant that components sum to the original source values. This ensures consistency between the operational forecast and the component breakdown.

**Extensibility**
   The :class:`~openstef_models.models.component_splitting.ComponentSplitter` abstract base class defines a minimal interface (``fit``, ``predict``, ``config``, ``is_fitted``). Custom splitters — for example, one using installed capacity registries or real-time curtailment signals — can be implemented by subclassing this interface.

When Not to Use Component Splitting
------------------------------------

Component splitting is the wrong tool if you need:

- **Accurate per-component forecasts** — if you have separate metering for solar and wind, forecast them individually using the standard forecasting pipeline (see :doc:`models`).
- **Behind-the-meter disaggregation** — component splitting assumes you know the *types* of generation present; it does not discover unknown loads.
- **Real-time control signals** — the output is informational, not a control setpoint.

Relationship to Other Concepts
------------------------------

- **Forecasting models** (:doc:`models`) produce the aggregated forecast that serves as input to component splitting.
- **BEAM** (:doc:`beam`) orchestrates the full forecasting workflow; component splitting can be a post-processing step within that pipeline.
- **Metalearning** (:doc:`metalearning`) selects the best model for the aggregated forecast — it operates upstream of component splitting.