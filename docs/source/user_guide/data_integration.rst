Data Integration
================

This page covers how to integrate OpenSTEF with your data infrastructure—reading time series from various storage backends, writing forecasts back, handling missing data, and validating inputs before they enter the forecasting pipeline.

OpenSTEF's core data abstraction is the ``TimeSeriesDataset``, a validated wrapper around pandas DataFrames with time series semantics. Your integration layer is responsible for loading raw data into this format and persisting forecast outputs.

.. mermaid:: /diagrams/user_guide/data_integration_diagram_1.mmd

Creating TimeSeriesDatasets from External Sources
-------------------------------------------------

All data entering OpenSTEF must be converted into a ``TimeSeriesDataset``. The dataset expects a pandas DataFrame with a ``DatetimeIndex`` and a consistent sample interval.

.. code-block:: python

   from datetime import timedelta
   import pandas as pd
   from openstef_core.datasets import TimeSeriesDataset

   # Load data from any source into a pandas DataFrame
   df = pd.read_csv("measurements.csv", parse_dates=["timestamp"], index_col="timestamp")

   # Wrap in a TimeSeriesDataset
   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=timedelta(minutes=15),
   )


Reading from PostgreSQL
^^^^^^^^^^^^^^^^^^^^^^^

A typical pattern for loading historical measurements from a relational database:

.. code-block:: python

   import pandas as pd
   from datetime import timedelta
   from sqlalchemy import create_engine
   from openstef_core.datasets import TimeSeriesDataset

   engine = create_engine("postgresql://user:pass@host:5432/energy_db")

   query = """
       SELECT timestamp, load, wind_speed, temperature
       FROM measurements
       WHERE station_id = 'substation_42'
         AND timestamp >= '2024-01-01'
       ORDER BY timestamp
   """

   df = pd.read_sql(query, engine, index_col="timestamp", parse_dates=["timestamp"])

   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=timedelta(minutes=15),
   )


Reading from InfluxDB
^^^^^^^^^^^^^^^^^^^^^

For time series databases like InfluxDB, use the client library to query and convert to a DataFrame:

.. code-block:: python

   from influxdb_client import InfluxDBClient
   from datetime import timedelta
   from openstef_core.datasets import TimeSeriesDataset

   client = InfluxDBClient(url="http://localhost:8086", token="my-token", org="my-org")
   query_api = client.query_api()

   flux_query = '''
       from(bucket: "energy")
           |> range(start: -90d)
           |> filter(fn: (r) => r._measurement == "load")
           |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
   '''

   df = query_api.query_data_frame(flux_query)
   df = df.set_index("_time").drop(columns=["result", "table", "_start", "_stop", "_measurement"])
   df.index.name = "timestamp"

   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=timedelta(minutes=15),
   )


Reading from S3
^^^^^^^^^^^^^^^

OpenSTEF includes S3 integration for benchmark storage. For custom data loading from S3, use a similar pattern:

.. code-block:: python

   import pandas as pd
   from datetime import timedelta
   from openstef_core.datasets import TimeSeriesDataset

   # Read parquet files directly from S3
   df = pd.read_parquet("s3://my-bucket/measurements/2024/load_data.parquet")

   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=timedelta(hours=1),
   )

.. note::

   For S3 access, ensure ``s3fs`` is installed and AWS credentials are configured via environment variables or ``~/.aws/credentials``.


Reading from Databricks
^^^^^^^^^^^^^^^^^^^^^^^

When running in a Databricks environment, use Spark to load data and convert to pandas:

.. code-block:: python

   from datetime import timedelta
   from openstef_core.datasets import TimeSeriesDataset

   # In a Databricks notebook or job
   spark_df = spark.sql("""
       SELECT timestamp, load, temperature, wind_speed
       FROM catalog.schema.measurements
       WHERE station_id = 'substation_42'
   """)

   df = spark_df.toPandas().set_index("timestamp").sort_index()

   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=timedelta(minutes=15),
   )


Writing Custom Data Sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production systems, encapsulate data loading in a reusable class:

.. code-block:: python

   from datetime import datetime, timedelta
   import pandas as pd
   from openstef_core.datasets import TimeSeriesDataset

   class MeasurementLoader:
       """Load measurement data from your infrastructure."""

       def __init__(self, connection_string: str, sample_interval: timedelta):
           self.connection_string = connection_string
           self.sample_interval = sample_interval

       def load(self, station_id: str, start: datetime, end: datetime) -> TimeSeriesDataset:
           # Replace with your actual data retrieval logic
           df = self._fetch_data(station_id, start, end)
           return TimeSeriesDataset(
               data=df,
               sample_interval=self.sample_interval,
           )

       def _fetch_data(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
           # Your implementation here
           ...


Data Validation
---------------

OpenSTEF provides built-in validation utilities to catch data quality issues before they propagate through the pipeline.

Validating Required Columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``validate_required_columns`` to ensure your data contains the expected features:

.. code-block:: python

   from openstef_core.datasets.validation import validate_required_columns

   required = ["load", "temperature", "wind_speed"]
   validate_required_columns(df, required)  # Raises MissingColumnsError if columns are absent


Frequency Validation
^^^^^^^^^^^^^^^^^^^^

The ``TimeSeriesDataset`` can check that your data has a consistent frequency:

.. code-block:: python

   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=timedelta(minutes=15),
       check_frequency=True,  # Raises if gaps or irregular intervals detected
   )


Handling Missing Data
---------------------

Missing data is common in energy measurement systems due to sensor failures, communication outages, or maintenance windows. OpenSTEF handles missing values at multiple levels.

Identifying Gaps
^^^^^^^^^^^^^^^^

Before feeding data into the pipeline, inspect your dataset for gaps:

.. code-block:: python

   import pandas as pd

   # Check for NaN values per column
   missing_summary = df.isna().sum()
   print(missing_summary)

   # Check for gaps in the time index
   expected_freq = pd.Timedelta(minutes=15)
   time_diffs = df.index.to_series().diff()
   gaps = time_diffs[time_diffs > expected_freq]
   print(f"Found {len(gaps)} gaps in the time series")


Reindexing to Fill Gaps
^^^^^^^^^^^^^^^^^^^^^^^^

Ensure a complete time index before creating the dataset:

.. code-block:: python

   # Create a complete index at the expected frequency
   full_index = pd.date_range(
       start=df.index.min(),
       end=df.index.max(),
       freq="15min",
   )

   # Reindex — missing timestamps become NaN rows
   df = df.reindex(full_index)

.. warning::

   OpenSTEF's training pipeline drops rows with NaN targets automatically. If all your target values are NaN after preprocessing, an ``InsufficientlyCompleteError`` is raised. Ensure your data has sufficient non-null target observations.


Interpolation Strategies
^^^^^^^^^^^^^^^^^^^^^^^^

For small gaps in weather features, interpolation can be appropriate:

.. code-block:: python

   # Linear interpolation for short gaps (up to 2 hours)
   max_gap = 8  # 8 × 15min = 2 hours
   df["temperature"] = df["temperature"].interpolate(method="linear", limit=max_gap)
   df["wind_speed"] = df["wind_speed"].interpolate(method="linear", limit=max_gap)

   # Do NOT interpolate the target (load) — let the model handle it
   # The pipeline will drop NaN targets during training


Writing Forecasts Back to Storage
---------------------------------

After running a forecast, OpenSTEF produces a ``ForecastDataset`` containing quantile predictions. Extract the results and write them to your storage backend.

.. code-block:: python

   from openstef_core.datasets.validated_datasets import ForecastDataset

   # After running the forecast pipeline, you have a ForecastDataset
   # Extract quantile forecasts as a DataFrame
   forecast_df = forecast_dataset.quantiles_data()

   # Get the median forecast
   median = forecast_dataset.median_series()

   # Write to PostgreSQL
   forecast_df.to_sql("forecasts", engine, if_exists="append", index=True)

   # Or write to parquet on S3
   forecast_df.to_parquet("s3://my-bucket/forecasts/2024/station_42.parquet")


Complete Data Pipeline Example
------------------------------

Here is a realistic end-to-end example showing data loading, validation, and forecast output:

.. code-block:: python

   from datetime import datetime, timedelta
   import pandas as pd
   from sqlalchemy import create_engine
   from openstef_core.datasets import TimeSeriesDataset
   from openstef_core.datasets.validation import validate_required_columns

   # --- Configuration ---
   DB_URL = "postgresql://user:pass@host:5432/energy_db"
   STATION_ID = "substation_42"
   SAMPLE_INTERVAL = timedelta(minutes=15)
   REQUIRED_COLUMNS = ["load", "temperature", "wind_speed", "radiation"]

   # --- Load data ---
   engine = create_engine(DB_URL)
   df = pd.read_sql(
       f"SELECT * FROM measurements WHERE station_id = '{STATION_ID}' ORDER BY timestamp",
       engine,
       index_col="timestamp",
       parse_dates=["timestamp"],
   )

   # --- Validate ---
   validate_required_columns(df, REQUIRED_COLUMNS)

   # --- Handle gaps ---
   full_index = pd.date_range(df.index.min(), df.index.max(), freq="15min")
   df = df.reindex(full_index)

   # Interpolate weather features only
   weather_cols = ["temperature", "wind_speed", "radiation"]
   df[weather_cols] = df[weather_cols].interpolate(method="linear", limit=8)

   # --- Create dataset ---
   dataset = TimeSeriesDataset(
       data=df,
       sample_interval=SAMPLE_INTERVAL,
   )

   # --- Use dataset in forecasting pipeline ---
   # See deployment page for full pipeline orchestration

.. mermaid:: /diagrams/user_guide/data_integration_diagram_2.mmd

Best Practices
--------------

- **Separate concerns**: Keep data loading logic independent from forecasting logic. Use dedicated loader classes.
- **Validate early**: Check column presence and data types before creating ``TimeSeriesDataset`` objects.
- **Don't interpolate targets**: Let OpenSTEF's pipeline handle missing target values—it drops NaN targets during training.
- **Use consistent time zones**: Ensure all timestamps are timezone-aware and in a consistent zone (UTC recommended).
- **Log data quality metrics**: Track the percentage of missing values per column to detect degrading data sources.
- **Batch writes**: When writing forecasts, batch inserts for performance rather than writing row-by-row.

For production deployment patterns including scheduling and orchestration, see :doc:`deployment`. For information on the forecasting pipeline itself, refer to the API documentation for ``openstef_core.datasets``.