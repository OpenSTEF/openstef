The openstef_core Package
=========================

This page covers the ``openstef_core`` package — the foundational layer of OpenSTEF that defines the dataset hierarchy, domain types, mixin system, and base classes used by all other packages. Understanding ``openstef_core`` is essential for working with any part of the OpenSTEF ecosystem.

.. mermaid:: /diagrams/architecture/core_diagram_1.mmd

Design Philosophy
-----------------

The core package follows two key principles:

- **Validated data at boundaries** — Domain-specific dataset classes enforce constraints (required columns, valid ranges) at construction time, so downstream code can trust its inputs.
- **Composable mixins** — Shared behavior (serialization, metadata, time series operations) lives in mixins that both ``TimeSeriesDataset`` and ``VersionedTimeSeriesDataset`` inherit independently.

The Mixin System
----------------

All dataset classes are built from two mixins defined in ``openstef_core.datasets.mixins``:

``TimeSeriesMixin``
   Provides time-aware operations: slicing by datetime range, frequency validation, horizon filtering, and alignment utilities.

``DatasetMixin``
   Provides persistence and metadata: save/load to Parquet, copy semantics, feature selection, and DataFrame transformation wrappers.

Both ``TimeSeriesDataset`` and ``VersionedTimeSeriesDataset`` inherit from these mixins directly, making them siblings rather than parent-child:

.. code-block:: python

   from openstef_core.datasets.mixins import DatasetMixin, TimeSeriesMixin

   class TimeSeriesDataset(TimeSeriesMixin, DatasetMixin):
       """Single-part time series with regular sampling."""
       ...

   class VersionedTimeSeriesDataset(TimeSeriesMixin, DatasetMixin):
       """Multi-part composition tracking data availability over time."""
       ...

This design means you can rely on the same interface (slicing, persistence, metadata) regardless of whether you hold a simple dataset or a versioned one.

Dataset Hierarchy
-----------------

TimeSeriesDataset
^^^^^^^^^^^^^^^^^

The primary data container in OpenSTEF. It wraps a pandas DataFrame with a consistent sampling interval and optional versioning through horizon or ``available_at`` columns.

.. code-block:: python

   from openstef_core.datasets import TimeSeriesDataset

   # Load from a Parquet file
   dataset = TimeSeriesDataset.from_parquet("weather_features.parquet")

   # Key operations
   subset = dataset.restrict_to_horizons(horizons)
   transformed = dataset.apply(lambda df: df.dropna())
   selected = dataset.select_features(["temperature", "wind_speed"])

Key properties and methods:

- ``sample_interval`` — the regular time step between observations
- ``copy_with(data)`` — create a new dataset with different data but same metadata
- ``frequency_matches(index)`` — verify a DatetimeIndex matches the dataset frequency

VersionedTimeSeriesDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Composes multiple ``TimeSeriesDataset`` instances into a unified dataset that tracks *when* data became available. This is critical for realistic backtesting — you can reconstruct exactly what information was known at any historical point.

.. code-block:: python

   from openstef_core.datasets import VersionedTimeSeriesDataset

   # Combine multiple data parts (e.g., successive weather forecast updates)
   versioned = VersionedTimeSeriesDataset.from_parts(
       parts=[forecast_v1, forecast_v2, forecast_v3],
   )

The class validates that parts have disjoint columns and matching sample intervals via ``validate_disjoint_columns`` and ``validate_same_sample_intervals``.

ForecastInputDataset
^^^^^^^^^^^^^^^^^^^^

A validated ``TimeSeriesDataset`` that guarantees a target column exists. Used as input to training and prediction pipelines.

.. code-block:: python

   from openstef_core.datasets import ForecastInputDataset, TimeSeriesDataset

   # Promote a generic dataset to a forecast input
   raw = TimeSeriesDataset.from_parquet("training_data.parquet")
   forecast_input = ForecastInputDataset.from_timeseries(raw, target_column="load_mw")

   # Access validated properties
   target = forecast_input.target_series()
   features = forecast_input.input_data()
   start = forecast_input.forecast_start()

ForecastDataset
^^^^^^^^^^^^^^^

Extends ``TimeSeriesDataset`` with validation for forecast outputs — ensures quantile columns and point forecasts are present and correctly structured.

EnsembleForecastDataset
^^^^^^^^^^^^^^^^^^^^^^^

Specialized for ensemble model outputs. Columns follow the naming convention ``<model>__<quantile>`` (separated by ``ENSEMBLE_COLUMN_SEP = "__"``), enabling structured access to individual ensemble member predictions.

EnergyComponentDataset
^^^^^^^^^^^^^^^^^^^^^^

Validates that the dataset contains columns matching known ``EnergyComponentType`` values (solar, wind, etc.), used in energy disaggregation workflows.

Domain Types
------------

The ``openstef_core.types`` module defines typed wrappers that enforce correctness and provide consistent serialization throughout the pipeline.

LeadTime
^^^^^^^^

Wraps a ``timedelta`` with ISO 8601 string serialization. Used everywhere a forecast horizon is specified.

.. code-block:: python

   from openstef_core.types import LeadTime

   # Create from timedelta
   horizon = LeadTime(timedelta(hours=24))

   # Convert to hours for display
   print(horizon.to_hours())  # 24.0

AvailableAt
^^^^^^^^^^^

Represents when data becomes available relative to a reference day, using the format ``DnTHHMM[tz]`` (e.g., ``D0T0900CET`` means "available at 09:00 CET on the reference day").

.. code-block:: python

   from openstef_core.types import AvailableAt
   from datetime import datetime

   availability = AvailableAt.from_string("D1T0600CET")

   # Apply to a specific date to get the absolute timestamp
   absolute_time = availability.apply(datetime(2024, 1, 15))

This type is central to versioned datasets — it defines the moment a data version becomes usable.

Quantile
^^^^^^^^

A ``float`` subclass constrained to ``[0, 1]`` with formatting and parsing utilities:

.. code-block:: python

   from openstef_core.types import Quantile

   q = Quantile(0.95)
   print(q.format())           # "q95" or similar string representation
   print(q.complementary())    # Quantile(0.05)
   print(q.to_percentile())    # 95.0

   # Parse from string
   parsed = Quantile.parse("q10")

EnergyComponentType
^^^^^^^^^^^^^^^^^^^

A ``StrEnum`` enumerating recognized energy component types (solar, wind, load, etc.). Used by ``EnergyComponentDataset`` for column validation.

Validation System
-----------------

The ``openstef_core.datasets.validation`` module provides reusable validators that dataset classes compose:

- ``validate_datetime_column`` — ensures a column contains valid datetime values
- ``validate_timedelta_column`` — ensures a column contains valid timedelta values
- ``validate_required_columns`` — raises ``MissingColumnsError`` if expected columns are absent
- ``validate_disjoint_columns`` — ensures multiple datasets don't share column names
- ``validate_same_sample_intervals`` — ensures datasets can be safely combined

These validators raise ``TimeSeriesValidationError`` (or subclasses like ``MissingColumnsError``) with descriptive messages, making debugging straightforward.

Utility Modules
---------------

The core package includes several utility modules:

- ``openstef_core.utils.datetime`` — timestamp alignment (``align_datetime``, ``align_datetime_to_time``)
- ``openstef_core.utils.pandas`` — optimized pandas operations (``combine_timeseries_indexes``, ``unsafe_sorted_range_slice_idxs``)
- ``openstef_core.utils.invariants`` — runtime assertions (``not_none``)
- ``openstef_core.utils`` — ISO timedelta conversion (``timedelta_from_isoformat``, ``timedelta_to_isoformat``)

How Core Supports Other Packages
--------------------------------

Every other OpenSTEF package builds on ``openstef_core``:

- **openstef_models** — Transforms and model wrappers consume and produce ``TimeSeriesDataset`` and ``ForecastDataset`` instances. See :doc:`models`.
- **openstef_beam** — Pipeline steps accept ``ForecastInputDataset`` for training and emit ``EnsembleForecastDataset`` from ensemble pipelines. See :doc:`beam`.
- **openstef_meta** — Ensemble forecasting models combine outputs using ``EnsembleForecastDataset`` and ``VersionedTimeSeriesDataset`` for availability-aware composition. See :doc:`meta`.

The strict typing and validation in core means that integration bugs surface immediately at dataset construction rather than deep inside pipeline execution.

.. note::

   The ``PydanticStringPrimitive`` base class (used by ``LeadTime`` and ``AvailableAt``) integrates with Pydantic v2 for automatic serialization in configuration models and API schemas.