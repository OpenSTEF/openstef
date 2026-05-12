Energy Component Splitting
==========================

Energy component splitting is the process of decomposing an aggregate load measurement into its constituent energy sources—such as solar generation, wind generation, and residual ("other") load. This page explains why component splitting is useful, how the ``ComponentSplitter`` interface works, and what implementations OpenSTEF provides.

Why Split Energy Components?
----------------------------

Grid operators and energy traders often measure only the *net load* at a substation or connection point. This single measurement is the sum of consumption, solar generation, wind generation, and potentially other distributed sources. Decomposing this aggregate signal is valuable for several reasons:

- **Improved forecasting accuracy** — Forecasting individual components (solar, wind, residual demand) separately and recombining them often outperforms forecasting the aggregate directly, because each component responds to different weather drivers. See :doc:`forecasting_basics` for more on the forecasting pipeline.
- **Grid planning and congestion management** — Understanding how much solar or wind capacity is behind a meter helps operators anticipate congestion under different weather scenarios.
- **Market settlement** — Some market designs require visibility into generation by source type, even when only aggregate measurements are available.
- **Feature engineering** — Decomposed components can serve as additional features for downstream models. See :doc:`feature_engineering` for details on predictors used in OpenSTEF.

.. mermaid:: /diagrams/concepts/component_splitting_diagram_1.mmd

The ComponentSplitter Interface
-------------------------------

All component splitters in OpenSTEF implement the abstract ``ComponentSplitter`` base class, which follows the ``Predictor`` protocol pattern used throughout the library. The interface guarantees a consistent contract regardless of the splitting method used.

.. code-block:: python

   from openstef_models.models.component_splitting.component_splitter import (
       ComponentSplitter,
       ComponentSplitterConfig,
   )
   from openstef_core.datasets import TimeSeriesDataset, EnergyComponentDataset
   from openstef_core.types import EnergyComponentType

The base configuration defines two parameters shared by all splitters:

- ``source_column`` — The column in the input dataset representing the total load to split (default: ``"load"``).
- ``components`` — The list of energy component types to produce (default: all defined ``EnergyComponentType`` values).

Every ``ComponentSplitter`` must implement:

- ``config`` — Property returning the splitter's configuration.
- ``fit(data, data_val=None)`` — Train the splitter (may be a no-op for rule-based methods).
- ``predict(data)`` — Perform the actual splitting, returning an ``EnergyComponentDataset``.
- ``is_fitted`` — Property indicating whether the splitter is ready for prediction.

A key invariant: the predicted components must sum to the original source column values. This ensures physical consistency of the decomposition.

EnergyComponentDataset
^^^^^^^^^^^^^^^^^^^^^^

The output of any component splitter is an ``EnergyComponentDataset``, a specialized time series dataset that guarantees the presence of columns for all energy component types (``wind``, ``solar``, ``other``):

.. code-block:: python

   from openstef_core.datasets import EnergyComponentDataset
   from datetime import timedelta
   import pandas as pd

   energy_data = pd.DataFrame({
       "wind": [50, 60],
       "solar": [30, 40],
       "other": [20, 25],
   }, index=pd.date_range("2025-01-01", periods=2, freq="h"))

   dataset = EnergyComponentDataset(energy_data, timedelta(hours=1))
   print(dataset.feature_names)  # ['wind', 'solar', 'other']

Available Implementations
-------------------------

OpenSTEF ships with two component splitter implementations, ranging from a simple baseline to a weather-driven model.

Constant Component Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ConstantComponentSplitter`` applies fixed, user-defined ratios to divide the total load into components. It requires no training and is useful when:

- You already know the approximate energy mix at a location (e.g., a connection point with 60% solar and 40% wind capacity).
- You need a simple baseline to compare against more sophisticated methods.

.. code-block:: python

   from openstef_models.models.component_splitting.constant_component_splitter import (
       ConstantComponentSplitter,
       ConstantComponentSplitterConfig,
   )
   from openstef_core.types import EnergyComponentType

   config = ConstantComponentSplitterConfig(
       source_column="load",
       component_ratios={
           EnergyComponentType.SOLAR: 0.6,
           EnergyComponentType.WIND: 0.4,
       },
   )

   splitter = ConstantComponentSplitter(config)
   components = splitter.predict(time_series_data)

The class also provides convenient factory methods for common configurations:

.. code-block:: python

   # Pre-configured for a pure solar park
   solar_splitter = ConstantComponentSplitter.known_solar_park()

   # Pre-configured for a pure wind farm
   wind_splitter = ConstantComponentSplitter.known_wind_farm()

.. note::

   The constant splitter is deterministic and does not account for weather conditions. It works well for locations with a single dominant generation type but poorly for mixed portfolios where the ratio varies with weather.

Linear Component Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``LinearComponentSplitter`` uses a pre-trained linear model that incorporates weather features—specifically solar radiation and wind speed at 100m—to estimate the contribution of each component. It splits the load into three predefined components:

- **Wind on shore**
- **Solar**
- **Other** (residual load)

.. code-block:: python

   from openstef_models.models.component_splitting.linear_component_splitter import (
       LinearComponentSplitter,
       LinearComponentSplitterConfig,
   )

   config = LinearComponentSplitterConfig(
       source_column="load",
       radiation_column="radiation",
       windspeed_100m_column="windspeed_100m",
   )

   splitter = LinearComponentSplitter(config)
   # The model ships pre-trained; no fit() call required
   components = splitter.predict(time_series_data)

The linear splitter ships with a pre-trained model (from OpenSTEF V3.4.24) and does not currently support re-training via ``fit()``. The input ``TimeSeriesDataset`` must contain the configured radiation and wind speed columns alongside the load column.

.. warning::

   The linear splitter requires weather data (radiation and wind speed) in the input dataset. If these columns are missing, prediction will fail. Ensure your data pipeline includes weather features before calling ``predict()``.

.. note:: [VISUALIZATION: Bar chart comparing component splitting results from constant vs. linear splitter across a 24-hour period, showing how the linear splitter captures diurnal solar patterns while the constant splitter produces flat ratios]

Choosing a Splitter
-------------------

+----------------------------+-------------------+---------------------+-------------------+
| Criterion                  | Constant          | Linear              |                   |
+============================+===================+=====================+===================+
| Weather data required      | No                | Yes                 |                   |
+----------------------------+-------------------+---------------------+-------------------+
| Training required          | No                | No (pre-trained)    |                   |
+----------------------------+-------------------+---------------------+-------------------+
| Captures diurnal patterns  | No                | Yes                 |                   |
+----------------------------+-------------------+---------------------+-------------------+
| Best for                   | Known single-type | Mixed portfolios    |                   |
|                            | installations     | with weather data   |                   |
+----------------------------+-------------------+---------------------+-------------------+

For locations where you have reliable weather forecasts and a mix of generation types, the linear splitter will produce more physically realistic decompositions. For simple cases—a dedicated solar park or wind farm—the constant splitter with appropriate ratios is sufficient and avoids the weather data dependency.

Implementing a Custom Splitter
------------------------------

You can create your own component splitter by subclassing ``ComponentSplitter``:

.. code-block:: python

   from openstef_models.models.component_splitting.component_splitter import (
       ComponentSplitter,
       ComponentSplitterConfig,
   )
   from openstef_core.datasets import EnergyComponentDataset, TimeSeriesDataset


   class MyCustomSplitterConfig(ComponentSplitterConfig):
       """Configuration for your custom splitter."""
       my_parameter: float = 0.5


   class MyCustomSplitter(ComponentSplitter):
       """A custom component splitter implementation."""

       def __init__(self, config: MyCustomSplitterConfig) -> None:
           super().__init__()
           self._config = config
           self._fitted = False

       @property
       def config(self) -> MyCustomSplitterConfig:
           return self._config

       @property
       def is_fitted(self) -> bool:
           return self._fitted

       def fit(self, data: TimeSeriesDataset, data_val=None) -> None:
           # Your training logic here
           self._fitted = True

       def predict(self, data: TimeSeriesDataset) -> EnergyComponentDataset:
           # Your splitting logic here
           # Ensure components sum to the source column
           ...

The key constraint to respect: the output components must sum to the original source column values. This physical consistency guarantee is what allows downstream forecasting pipelines to trust the decomposition.

Related Topics
--------------

- :doc:`forecasting_basics` — How OpenSTEF uses component forecasts in the overall pipeline
- :doc:`feature_engineering` — Weather features (radiation, wind speed) used by the linear splitter
- :doc:`reliability_and_fallback` — Fallback strategies when component splitting or forecasting fails