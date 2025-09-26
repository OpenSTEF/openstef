.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

MedianForecaster
================

The :class:`~openstef_models.models.forecasting.median_forecaster.MedianForecaster` is an autoregressive forecasting model that uses the median of historical lag features to predict future values. It's particularly useful for signals with slow dynamics compared to the sampling rate or signals that switch between stable states.

Overview
--------

The MedianForecaster computes the median of lag features (e.g., T-60min, T-120min, T-180min) to make predictions. It includes autoregressive capabilities where the model's own predictions are used as lag features for subsequent time steps.

Key Features
~~~~~~~~~~~~

* **Autoregressive**: Uses previous predictions as inputs for future predictions
* **Robust to noise**: Median computation naturally filters out outliers
* **No training required**: Directly computes median from available lag features
* **Handles missing data**: Gracefully ignores NaN values in median calculation
* **State preservation**: Supports serialization and restoration of model state

When to Use
~~~~~~~~~~~

The MedianForecaster is suitable for:

* Signals with very slow dynamics compared to the sampling rate, possibly with noise
* Signals that switch between stable states (e.g., waste heat from industrial processes)
* Scenarios where you want a simple, robust baseline model
* Cases where limited historical data is available for training

Configuration
-------------

The MedianForecaster uses :class:`~openstef_models.models.forecasting.median_forecaster.MedianForecasterConfig` for configuration:

.. code-block:: python

    from openstef_models.models.forecasting.median_forecaster import (
        MedianForecaster, 
        MedianForecasterConfig
    )
    from openstef_core.types import LeadTime, Quantile
    from datetime import timedelta

    config = MedianForecasterConfig(
        quantiles=[Quantile(0.5)],  # Only median (0.5) quantile supported
        horizons=[LeadTime(timedelta(hours=6))],  # Single horizon
    )

Important Notes
~~~~~~~~~~~~~~~

* **Quantiles**: Only supports median (quantile 0.5) predictions
* **Horizons**: Designed for single-horizon forecasting
* **Lag Features**: Requires evenly spaced lag features (T-60min, T-120min, etc.)

Data Requirements
-----------------

The MedianForecaster expects data with specific lag features:

Feature Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~~

Lag features must follow this naming pattern:

* **Minutes**: ``T-<lag_in_minutes>min`` (e.g., T-60min, T-120min)
* **Days**: ``T-<lag_in_days>d`` (e.g., T-1d, T-2d)

The lag features must be evenly spaced and match the frequency of the input data.

Example Data Structure
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Example data with hourly frequency and 60-minute lag intervals
    data = pd.DataFrame({
        "load": [1000, 1100, 1200, 1150, 1050],
        "T-60min": [np.nan, 1000, 1100, 1200, 1150],    # 1 hour lag
        "T-120min": [np.nan, np.nan, 1000, 1100, 1200], # 2 hour lag
        "T-180min": [np.nan, np.nan, np.nan, 1000, 1100], # 3 hour lag
        "temperature": [15, 16, 17, 18, 16],  # Other features allowed
    }, index=pd.date_range("2025-01-01", periods=5, freq="1h"))

Usage Example
-------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from openstef_core.datasets.validated_datasets import ForecastInputDataset
    from openstef_models.models.forecasting.median_forecaster import (
        MedianForecaster, 
        MedianForecasterConfig
    )
    from openstef_core.types import LeadTime, Quantile
    from datetime import timedelta

    # Create configuration
    config = MedianForecasterConfig(
        quantiles=[Quantile(0.5)],
        horizons=[LeadTime(timedelta(hours=6))],
    )

    # Create forecaster
    forecaster = MedianForecaster(config)

    # Prepare data (assuming 'data' DataFrame exists with lag features)
    dataset = ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime(2025, 1, 1, 12, 0, 0),
    )

    # Fit and predict
    forecaster.fit(dataset)
    predictions = forecaster.predict(dataset)

    # Access results
    median_predictions = predictions.data["quantile_P50"]

State Management
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save model state
    state = forecaster.to_state()

    # Create new forecaster from state
    new_forecaster = MedianForecaster()
    restored_forecaster = new_forecaster.from_state(state)

    # Verify restoration
    assert restored_forecaster.is_fitted

Algorithm Details
-----------------

The MedianForecaster implements the following algorithm:

1. **Validation**: Extract and validate lag features from input data
2. **Frequency Check**: Ensure data frequency matches lag feature spacing
3. **Reindexing**: Fill gaps in time series to ensure continuity
4. **Median Calculation**: For each time step, compute median of available lag features
5. **Autoregression**: Update future lag values with current predictions
6. **Output Filtering**: Return predictions only after forecast_start (if specified)

Autoregressive Behavior
~~~~~~~~~~~~~~~~~~~~~~~

The autoregressive component works as follows:

* When a median is computed for time step ``t``, this value is used to fill lag features for future time steps
* Future predictions at time ``t+k`` will use this predicted value as one of their lag inputs
* This creates a dependency where later predictions are influenced by earlier ones

This behavior is particularly useful when the prediction horizon extends beyond the available lag feature window.

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

* Use evenly spaced lag features matching your data frequency
* Ensure lag features are named correctly (T-60min, T-120min, etc.)
* Include sufficient lag history for meaningful median calculation

Model Selection
~~~~~~~~~~~~~~~

* Use MedianForecaster for stable, slow-changing signals
* Consider it as a robust baseline before trying more complex models
* Suitable when training data is limited or when interpretability is important

Performance Tips
~~~~~~~~~~~~~~~~

* Use small training datasets - the model doesn't learn parameters
* Set completeness_threshold to 0 to allow missing data
* Use only one training horizon to avoid confusion

Limitations
-----------

* **Single Quantile**: Only supports median (0.5) predictions
* **Single Horizon**: Not designed for multi-horizon forecasting
* **Lag Dependency**: Requires properly engineered lag features
* **Simple Logic**: May not capture complex patterns or relationships

For more complex forecasting needs, consider other OpenSTEF models like XGBoost-based forecasters.

API Reference
-------------

.. currentmodule:: openstef_models.models.forecasting.median_forecaster

.. autoclass:: MedianForecaster
   :members:
   :inherited-members:

.. autoclass:: MedianForecasterConfig
   :members:

.. autoclass:: MedianForecasterHyperParams
   :members: