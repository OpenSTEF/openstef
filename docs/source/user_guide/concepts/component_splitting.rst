.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _concept_component_splitting:

Component Splitting
===================

Component splitting is the process of decomposing an aggregated energy forecast into
its constituent parts: solar generation, wind generation, and residual (base) load.
Grid operators typically observe only the net load at a substation or connection point,
but operational decisions often require understanding *what is underneath* that
aggregated signal.

This page explains why component splitting matters, how OpenSTEF approaches it, and
what invariants the system enforces.

Why Decompose a Forecast?
--------------------------

A grid operator's measurement infrastructure records the net power flow at a
connection point. This single time series is the sum of many underlying processes:
rooftop solar feeding back into the grid (negative contribution), wind turbines
generating power, heat pumps cycling, industrial machinery running, and residential
consumption patterns overlapping.

Forecasting this aggregated load directly is statistically sound; the aggregated
signal is smoother and more predictable than its parts. However, once a forecast
exists, operators need to decompose it for several reasons:

- **Congestion analysis**: Understanding how much solar generation is expected helps
  predict reverse power flow and transformer loading.
- **Grid planning**: Capacity studies require knowledge of generation mix to assess
  future infrastructure needs.
- **Regulatory reporting**: Some jurisdictions require reporting of renewable
  generation volumes separately from consumption.
- **Market operations**: Balancing responsible parties may need component-level
  forecasts to settle imbalances by source.
- **Curtailment decisions**: When congestion is anticipated, operators need to know
  which generation source to curtail and by how much.

.. note::

   Component splitting does **not** improve forecast accuracy. The forecast is
   produced first on the aggregated load signal, then decomposed afterward. The
   decomposition is a post-processing step that redistributes the forecasted total
   into parts.

Forecast First, Split Second
-----------------------------

OpenSTEF's architecture enforces a clear separation between forecasting and
splitting. The forecasting pipeline (see :doc:`/user_guide/guides/forecasting`)
produces a prediction of total load. Component splitting then takes that prediction
and allocates it across energy components using weather features and known physical
relationships.

.. mermaid:: /diagrams/user_guide/concepts/component_splitting_diagram_1.mmd

This two-stage approach has important implications:

- The sum of all components always equals the original forecast (an invariant
  enforced by the splitting algorithm).
- Errors in the aggregated forecast propagate into all components proportionally.
- Improving the splitting model does not change the total forecast; it only
  redistributes the same total differently.

Available Splitter Implementations
-----------------------------------

OpenSTEF provides multiple component splitting strategies through a common interface
defined by :class:`~openstef_models.models.component_splitting.ComponentSplitter`.

.. list-table:: Component Splitter Implementations
   :header-rows: 1
   :widths: 25 40 35

   * - Splitter
     - Approach
     - When to Use
   * - :class:`~openstef_models.models.component_splitting.linear_component_splitter.LinearComponentSplitter`
     - Pre-trained linear model using radiation and wind speed features to estimate
       solar and wind contributions; residual becomes "other"
     - Default choice when weather data (radiation, wind speed at 100m) is available
   * - :class:`~openstef_models.models.component_splitting.constant_component_splitter.ConstantComponentSplitter`
     - Splits load using fixed ratios per component (e.g., 60% solar, 40% wind)
     - Quick approximation when weather features are unavailable or for testing

All splitters share the same contract: they accept a :class:`~openstef_core.datasets.TimeSeriesDataset`
containing the total load column and return an :class:`~openstef_core.datasets.EnergyComponentDataset`
whose columns sum to the original source values.

Energy Component Types
^^^^^^^^^^^^^^^^^^^^^^

The system defines a fixed set of recognized component types through
:class:`~openstef_core.types.EnergyComponentType`:

- **Solar**: Photovoltaic generation (represented as negative values, since generation
  reduces net load)
- **Wind**: Wind turbine generation (also negative values)
- **Other**: The residual after solar and wind are subtracted; encompasses base
  industrial load, residential consumption, and any unmodeled generation

The Pipeline
------------

The high-level orchestration is handled by
:class:`~openstef_models.models.component_splitting_model.ComponentSplittingModel`,
which composes three stages:

1. **Preprocessing**: A transform pipeline that prepares raw input data (e.g.,
   selecting and renaming columns, handling missing values).
2. **Splitting**: The core algorithm (one of the splitter implementations above) that
   performs the actual decomposition.
3. **Postprocessing**: A transform pipeline that adjusts the output (e.g., clipping,
   scaling, or formatting).

This composition follows the same pattern used elsewhere in OpenSTEF's model
architecture (see :doc:`/user_guide/concepts/models`).

.. code-block:: python

   model = ComponentSplittingModel(
       component_splitter=splitter,
       preprocessing=preprocessing_pipeline,
       source_column="load",
   )
   components = model.predict(forecast_data)

The ``source_column`` parameter tells the model which column in the input dataset
contains the total load to decompose.

How the Linear Splitter Works
------------------------------

The :class:`~openstef_models.models.component_splitting.linear_component_splitter.LinearComponentSplitter` uses a pre-trained linear model that maps weather
features to generation components:

- **Inputs**: radiation (solar irradiance) and wind speed at 100m height, plus the
  total load value.
- **Outputs**: Estimated solar and wind contributions (clipped to be non-positive,
  since generation reduces net load).
- **Residual**: The "other" component is computed as the difference between total load
  and the sum of solar and wind estimates.

This ensures the invariant holds: solar + wind + other = total load.

The linear model is shipped pre-trained (from OpenSTEF V3.4.24) and does not require
fitting on local data. This makes it immediately usable but means it represents
average relationships rather than site-specific ones.

Workflow Integration
---------------------

For production use, component splitting integrates into the broader workflow system
through :class:`~openstef_models.workflows.custom_component_split_workflow.CustomComponentSplitWorkflow`, which adds lifecycle callbacks for
monitoring, logging, and model management. This follows the same callback pattern as
the forecasting workflow described in :doc:`/user_guide/guides/deployment`.

Key Constraints and Assumptions
--------------------------------

When using component splitting, keep these constraints in mind:

- **Weather data required** (for the linear splitter): The input dataset must contain
  radiation and wind speed columns. Missing values are dropped before prediction.
- **Summation invariant**: Components always sum to the source column value. If the
  forecast is biased, all components inherit that bias.
- **Sign conventions**: Generation components (solar, wind) are negative values in the
  net load frame of reference. Consumption is positive.
- **Temporal resolution**: The splitter operates at whatever resolution the input data
  provides; it does not resample.
- **No feedback loop**: Splitting results do not feed back into the forecasting model.
  They are purely informational outputs for downstream consumers.

.. warning::

   If the aggregated forecast contains large errors (e.g., during extreme weather
   events), the component split will faithfully distribute those errors across
   components. Always evaluate forecast quality at the aggregated level first
   (see :doc:`/user_guide/concepts/beam`).

.. seealso::

   - :ref:`concept_models` for the model architecture that component splitting builds on.
   - :doc:`/user_guide/guides/forecasting` for how the aggregated forecast is produced before splitting.
   - :doc:`/user_guide/guides/deployment` for integrating component splitting into production workflows.