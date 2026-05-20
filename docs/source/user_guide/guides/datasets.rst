Datasets
========

Why DataFrames Aren't Enough
----------------------------

A raw ``pandas.DataFrame`` carries data but not the *context* the forecasting pipeline needs to use it correctly. Consider weather forecast data: the temperature prediction issued at 6:00 AM for noon tomorrow is a fundamentally different piece of information than the temperature prediction issued at 12:00 PM for the same target time. Both share the same index timestamp, but they were *available* at different moments.

Energy forecasting pipelines must answer questions like:

- What data was actually available when this prediction was made?
- What is the minimum lead time between data publication and the forecast target?
- How should duplicate observations for the same timestamp be resolved?

Plain DataFrames cannot answer these questions without ad-hoc column conventions that every function must re-implement. OpenSTEF's dataset types solve this by embedding temporal metadata directly into the data container.

.. mermaid:: /diagrams/user_guide/guides/datasets_diagram_1.mmd

TimeSeriesDataset
-----------------

:class:`~openstef.datasets.TimeSeriesDataset` wraps a pandas DataFrame with essential metadata:

- **Sample interval** — the expected time resolution (e.g., 15 minutes), enabling the pipeline to detect gaps and validate alignment.
- **Versioning columns** — optional ``available_at`` or ``horizon`` columns that mark when each row became available or what lead time it represents.
- **Feature names** — the subset of columns that are actual input features (excluding internal metadata columns).

A ``TimeSeriesDataset`` can be either *versioned* (has an ``available_at`` or ``horizon`` column) or *non-versioned* (a simple time series). The ``is_versioned`` property tells you which:

.. code-block:: python

   from openstef.datasets import TimeSeriesDataset
   from datetime import timedelta

   dataset = TimeSeriesDataset(df, sample_interval=timedelta(minutes=15))
   dataset.is_versioned  # True if available_at or horizon column present

The dataset exposes ``available_at_series`` and ``lead_time_series`` properties that derive availability and lead time information regardless of which column was provided in the source data.

VersionedTimeSeriesDataset
--------------------------

:class:`~openstef.datasets.VersionedTimeSeriesDataset` models the reality that forecasting inputs arrive from multiple sources with different temporal characteristics. Weather forecasts, price signals, and load measurements each have their own publication schedules and revision patterns.

**The core design insight:** naively concatenating versioned data from multiple sources creates an O(n²) space problem because each source's ``(timestamp, available_at)`` pairs don't align. ``VersionedTimeSeriesDataset`` solves this through **lazy composition** — it holds a list of ``TimeSeriesDataset`` parts and delays actual DataFrame concatenation until resolution methods are called.

.. code-block:: python

   from openstef.datasets import VersionedTimeSeriesDataset

   dataset = VersionedTimeSeriesDataset([weather_part, load_part])
   dataset.is_versioned  # True

All data parts must share the same sample interval and have disjoint feature sets. The combined index is the union of all part indices.

For simple cases where all data lives in a single DataFrame, use the convenience constructor:

.. code-block:: python

   dataset = VersionedTimeSeriesDataset.from_dataframe(df, timedelta(minutes=15))

Resolution Methods
------------------

Resolution methods answer the question: *"Given temporal constraints, which slice of data should I use?"* Each returns a new dataset instance (immutable pattern), enabling method chaining.

.. list-table:: Resolution Methods
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - ``filter_by_available_before(timestamp)``
     - Keep only data published before a specific moment. Essential for preventing lookahead bias in backtesting.
   * - ``filter_by_available_at(schedule)``
     - Filter by a daily availability schedule (e.g., ``'D-1T0600'`` means "yesterday's 6:00 AM publication"). Models operational data delivery patterns.
   * - ``filter_by_lead_time(lead_time)``
     - Keep only data with at least the specified advance notice between publication and target time.
   * - ``select_version()``
     - Collapse versioned data to a single ``TimeSeriesDataset`` by selecting the latest available version for each timestamp. This is the final step that produces a concrete, non-versioned result.

.. mermaid:: /diagrams/user_guide/guides/datasets_diagram_2.mmd

Preventing Lookahead Bias
^^^^^^^^^^^^^^^^^^^^^^^^^

The most critical use of resolution methods is ensuring that training and backtesting never use data that wouldn't have been available at prediction time:

.. code-block:: python

   # Only use data that was published before the prediction moment
   snapshot = dataset.filter_by_available_before(prediction_time).select_version()

This pattern is built into OpenSTEF's backtesting infrastructure. See :doc:`backtesting` for how this integrates with model evaluation.

Multi-Horizon Training with to_horizons()
------------------------------------------

Many forecasting scenarios require a single model trained across multiple forecast horizons (e.g., 1-hour ahead, 6-hours ahead, 24-hours ahead). The :meth:`~openstef.datasets.VersionedTimeSeriesDataset.to_horizons` method automates this:

.. code-block:: python

   training_data = dataset.to_horizons(horizons=[lead_time_1h, lead_time_6h, lead_time_24h])

Internally, ``to_horizons()`` chains resolution methods for each horizon:

1. ``filter_by_lead_time(horizon)`` — select data with sufficient lead time
2. ``select_version()`` — deduplicate to one row per timestamp
3. Assign a ``horizon`` column marking the lead time
4. Concatenate all horizons into a single ``TimeSeriesDataset``

The resulting dataset has a ``horizon`` column and can be further sliced with ``select_horizon()`` to retrieve data for a specific lead time.

Interoperability with pandas
----------------------------

OpenSTEF datasets are designed to work *with* pandas, not replace it. Key interoperability methods:

- ``to_pandas()`` — export to a DataFrame with metadata stored in ``df.attrs``
- ``from_pandas()`` — reconstruct a dataset from a DataFrame with attrs metadata
- ``to_parquet()`` / ``read_parquet()`` — persist to disk with full metadata
- ``pipe_pandas(func)`` — apply an arbitrary pandas transformation while preserving dataset metadata

This means you can drop into pandas for custom transformations without losing the temporal context:

.. code-block:: python

   cleaned = dataset.pipe_pandas(lambda df: df.dropna(subset=["load"]))

When to Use Which Type
----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Scenario
     - Recommended Type
   * - Simple historical load data (no revisions)
     - ``TimeSeriesDataset`` without versioning columns
   * - Weather forecasts with publication times
     - ``TimeSeriesDataset`` with ``available_at`` column, composed into ``VersionedTimeSeriesDataset``
   * - Multiple data sources with different schedules
     - ``VersionedTimeSeriesDataset`` with multiple parts
   * - Training data prepared for multiple horizons
     - Result of ``to_horizons()`` — a ``TimeSeriesDataset`` with ``horizon`` column

Related Topics
--------------

- :doc:`forecasting` — how datasets flow through the prediction pipeline
- :doc:`backtesting` — using resolution methods to prevent lookahead bias in evaluation
- :doc:`reliability_fallback` — handling missing data within datasets