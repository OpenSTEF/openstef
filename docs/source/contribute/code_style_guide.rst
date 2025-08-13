.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

.. _code_style_guide:

===========
Style Guide
===========

This guide outlines the coding standards and conventions used in OpenSTEF 4.0. 
Following these guidelines helps maintain consistency, readability, and maintainability 
across the codebase.

.. note::

    Most style issues are automatically handled by our development tools. Running 
    ``poe all`` will fix formatting and catch style violations automatically.

.. _code-formatting:

Code Formatting
===============

PEP 8 with Ruff
---------------

OpenSTEF follows `PEP 8 <https://peps.python.org/pep-0008/>`_ as enforced by 
`Ruff <https://docs.astral.sh/ruff/>`_. Our configuration extends the standard 
line length to **88 characters** (matching Black's default).

Key formatting rules:

* **Line length**: 88 characters maximum
* **Indentation**: 4 spaces (no tabs)
* **Quotes**: Prefer double quotes for strings
* **Trailing commas**: Required in multi-line constructs
* **Import sorting**: Alphabetical within sections

Check and fix formatting:

.. code-block:: bash

    # Check formatting without changes
    poe format --check

    # Auto-fix formatting issues
    poe format

.. _import-conventions:

Import Conventions
==================

Standard library imports
------------------------

Import standard library modules without abbreviation:

.. code-block:: python

    import os
    import sys
    import logging
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Union

Scientific computing imports
----------------------------

Use standard scipy ecosystem conventions:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.stats as stats

OpenSTEF imports
----------------

Import OpenSTEF modules using full names to maintain clarity:

.. code-block:: python

    # Good: Clear module structure
    from openstef.models.forecasting import XGBoostForecaster
    from openstef.feature_engineering.temporal_transforms import LagTransform
    from openstef.pipelines.training import TrainingPipeline

    # Avoid: Unclear module imports
    from openstef.models import *
    from openstef import XGBoostForecaster

Import organization
-------------------

Organize imports in this order with blank lines between sections:

.. code-block:: python

    # 1. Standard library
    import logging
    from pathlib import Path
    from typing import Optional

    # 2. Third-party packages
    import numpy as np
    import pandas as pd

    # 3. OpenSTEF modules
    from openstef.models.forecasting import LinearForecaster
    from openstef.feature_engineering import LagTransform

.. _variable-naming:

Variable Naming Conventions
===========================

General naming rules
--------------------

* **Functions and variables**: ``snake_case``
* **Classes**: ``PascalCase``
* **Constants**: ``UPPER_SNAKE_CASE``
* **Private members**: Prefix with single underscore ``_private_method``
* **Special methods**: Double underscores ``__special__`` (use sparingly)

Domain-specific naming
----------------------

Use these standard names for common objects in energy forecasting:

.. code-block:: python

    # Time series data
    data: pd.DataFrame           # General time series data
    forecast_data: pd.DataFrame  # Data specifically for forecasting
    weather_data: pd.DataFrame   # Weather-related data
    load_data: pd.DataFrame      # Energy load/consumption data

    # Models and predictors
    model: ForecastModel         # Trained forecasting model
    forecaster: BaseForecaster   # Model class instance
    pipeline: TrainingPipeline   # Training/inference pipeline

    # Predictions and evaluation
    y_true: np.ndarray          # Ground truth values
    y_pred: np.ndarray          # Model predictions
    forecast: pd.DataFrame       # Forecast output with uncertainty
    residuals: np.ndarray       # Prediction errors

    # Time-related variables
    horizon: int                # Forecast horizon in hours
    timestamp: pd.Timestamp     # Single point in time
    available_at: pd.Timestamp  # When data becomes available

Avoid generic names
-------------------

.. code-block:: python

    # Good: Descriptive names
    energy_consumption = load_sample_data()
    weather_features = extract_weather_features(data)
    mae_score = calculate_mae(y_true, y_pred)

    # Avoid: Generic names
    df = load_sample_data()
    result = extract_weather_features(data) 
    score = calculate_mae(y_true, y_pred)

.. _type-hints:

Type Hints
==========

OpenSTEF 4.0 uses comprehensive type hints for better code clarity and IDE support.

Required type hints
-------------------

All **public functions** and **class methods** must have type hints:

.. code-block:: python

    def train_forecaster(
        data: TimeseriesDataset, 
        config: ModelConfig,
        validation_split: float = 0.2
    ) -> TrainingResult:
        """Train a forecasting model with the given configuration."""

    class XGBoostForecaster:
        def __init__(self, config: XGBoostConfig) -> None:
            self.config = config

        def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostForecaster":
            # Implementation
            return self

Common type patterns
--------------------

Use these standard patterns for common OpenSTEF types:

.. code-block:: python

    from typing import Optional, Union, Dict, List, Any
    from pathlib import Path

    # File paths - prefer pathlib.Path
    def save_model(model: ForecastModel, path: Path) -> None:
        pass

    # Optional parameters with defaults
    def create_forecaster(
        model_type: str = "xgboost",
        config: Optional[ModelConfig] = None
    ) -> BaseForecaster:
        pass

    # Union types for flexible inputs
    def load_data(source: Union[str, Path, pd.DataFrame]) -> TimeseriesDataset:
        pass

    # Generic containers
    def validate_features(features: List[str]) -> Dict[str, bool]:
        pass

Type checking
-------------

Validate type hints using our development tools:

.. code-block:: bash

    # Type check with pyright
    poe type

    # Full validation including type checks
    poe all --check

.. _function-design:

Function and Class Design
=========================

Function structure
------------------

Design functions following the **Single Responsibility Principle**:

.. code-block:: python

    def calculate_weather_features(
        temperature: pd.Series,
        humidity: pd.Series,
        wind_speed: pd.Series
    ) -> pd.DataFrame:
        """Calculate derived weather features from basic measurements.
        
        Args:
            temperature: Temperature in Celsius.
            humidity: Relative humidity as percentage.
            wind_speed: Wind speed in m/s.
            
        Returns:
            DataFrame with derived features: heat_index, wind_chill, comfort_index.
        """
        # Single, focused responsibility
        heat_index = _calculate_heat_index(temperature, humidity)
        wind_chill = _calculate_wind_chill(temperature, wind_speed)
        comfort_index = _calculate_comfort_index(temperature, humidity)
        
        return pd.DataFrame({
            'heat_index': heat_index,
            'wind_chill': wind_chill,
            'comfort_index': comfort_index
        })

Parameter handling
------------------

Use keyword-only arguments for configuration parameters:

.. code-block:: python

    def train_model(
        data: TimeseriesDataset,
        *,  # Force keyword-only arguments after this
        validation_split: float = 0.2,
        random_state: Optional[int] = None,
        early_stopping: bool = True
    ) -> TrainingResult:
        """Forces explicit parameter names for better clarity."""

Class hierarchy design
----------------------

Follow OpenSTEF's **four-level hierarchy** (see the `design decisions document <https://lf-energy.atlassian.net/wiki/spaces/OS/pages/334037011/OpenSTEF+4.0+Proposal#Feature-Hierarchy-Convention>`_):

.. code-block:: python

    # Level 1: Pure functions
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Standalone utility function."""
        return np.mean(np.abs(y_true - y_pred))

    # Level 2: Standardized components  
    class LagTransform(BaseTransform):
        """Configurable component with fit/transform API."""
        
        def fit(self, data: TimeseriesDataset) -> "LagTransform":
            # Learn parameters if needed
            return self
            
        def transform(self, data: TimeseriesDataset) -> TimeseriesDataset:
            # Apply transformation
            pass

    # Level 3: Multi-component pipelines
    class TrainingPipeline:
        """Orchestrates multiple Level 2 components."""
        
        def __init__(
            self,
            transforms: List[BaseTransform],
            model: BaseForecaster
        ):
            self.transforms = transforms
            self.model = model

    # Level 4: Pre-configured solutions
    class StandardForecastingSolution:
        """Ready-to-use configuration for common use cases."""
        
        @classmethod
        def for_congestion_management(cls) -> "StandardForecastingSolution":
            """Optimized for peak detection accuracy."""
            pass

.. _docstring-standards:

Documentation Standards
=======================

All public functions and classes must have Google-style docstrings. See 
:doc:`document` for comprehensive documentation guidelines.

Minimal docstring requirements
------------------------------

.. code-block:: python

    def extract_lag_features(data: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Create lagged versions of input features.

        Args:
            data: Time series data with datetime index.
            lags: List of lag periods in hours.

        Returns:
            DataFrame with original and lagged features.
        """

Include examples for complex functions
--------------------------------------

.. code-block:: python

    def resample_timeseries(
        data: pd.DataFrame, 
        target_frequency: str = "1h"
    ) -> pd.DataFrame:
        """Resample time series data to target frequency.

        Args:
            data: Input time series with datetime index.
            target_frequency: Target frequency (e.g., '1h', '15min').

        Returns:
            Resampled data with target frequency.

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({
            ...     'load': [100, 120, 110]
            ... }, index=pd.date_range('2025-01-01', periods=3, freq='30min'))
            >>> hourly_data = resample_timeseries(data, '1h')
            >>> hourly_data.index.freq
            <Hour>
        """

.. _error-handling:

Error Handling and Logging
==========================

Exception handling
------------------

Use specific exception types and provide helpful error messages:

.. code-block:: python

    def validate_forecast_data(data: pd.DataFrame) -> None:
        """Validate input data for forecasting."""
        
        if data.empty:
            raise ValueError("Forecast data cannot be empty")
            
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                f"Data index must be DatetimeIndex, got {type(data.index)}"
            )
            
        required_columns = ['load', 'temperature']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

Custom exceptions
-----------------

Define domain-specific exceptions for clear error categorization:

.. code-block:: python

    class OpenSTEFError(Exception):
        """Base exception for OpenSTEF-specific errors."""
        pass

    class DataValidationError(OpenSTEFError):
        """Raised when input data fails validation."""
        pass

    class ModelTrainingError(OpenSTEFError):
        """Raised when model training fails."""
        pass

Logging practices
-----------------

Use Python's logging module for debug and info messages:

.. code-block:: python

    import logging

    _log = logging.getLogger(__name__)

    def train_forecaster(data: TimeseriesDataset) -> ForecastModel:
        """Train a forecasting model."""
        
        _log.info(f"Training model with {len(data)} samples")
        _log.debug(f"Feature columns: {data.feature_columns}")
        
        try:
            model = self._fit_model(data)
            _log.info("Model training completed successfully")
            return model
        except Exception as e:
            _log.error(f"Model training failed: {e}")
            raise ModelTrainingError(f"Training failed: {e}") from e

Logging guidelines:

* Use ``_log.debug()`` for detailed diagnostic information
* Use ``_log.info()`` for important operational information  
* Use ``_log.warning()`` for recoverable issues
* Use ``_log.error()`` for serious problems
* Avoid pre-computed f-strings in log messages for performance

.. _testing-guidelines:

Testing and Code Quality
========================

Test naming and structure
-------------------------

Write clear, descriptive test names:

.. code-block:: python

    def test_lag_transform_creates_correct_number_of_features():
        """Test that LagTransform creates expected number of lag features."""
        pass

    def test_forecaster_handles_missing_weather_data_gracefully():
        """Test forecaster behavior when weather data is incomplete."""
        pass

    def test_training_pipeline_raises_error_with_empty_dataset():
        """Test that TrainingPipeline raises appropriate error for empty data."""
        pass

Code quality checks
-------------------

Our development tools automatically enforce quality standards:

.. code-block:: bash

    # Run all quality checks
    poe all --check

    # Individual checks
    poe lint --check       # Ruff linting
    poe format --check     # Code formatting  
    poe type               # Type checking with pyright
    poe tests              # Run test suite
    poe doctests           # Test docstring examples

The ``poe all --check`` command must pass before any pull request can be merged.

.. _configuration-patterns:

Configuration and Data Patterns
===============================

Configuration classes
---------------------

Use dataclasses or Pydantic models for configuration:

.. code-block:: python

    from dataclasses import dataclass
    from typing import Optional, List

    @dataclass
    class ModelConfig:
        """Configuration for forecasting models."""
        
        model_type: str = "xgboost"
        learning_rate: float = 0.1
        max_depth: int = 6
        n_estimators: int = 100
        random_state: Optional[int] = None
        
        def __post_init__(self) -> None:
            """Validate configuration after initialization."""
            if self.learning_rate <= 0:
                raise ValueError("Learning rate must be positive")

Data validation patterns
------------------------

Implement data validation using clear, testable functions:

.. code-block:: python

    def validate_timeseries_data(data: pd.DataFrame) -> TimeseriesDataset:
        """Validate and convert DataFrame to TimeseriesDataset.
        
        Raises:
            DataValidationError: If data fails validation checks.
        """
        # Check data structure
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataValidationError("Index must be DatetimeIndex")
            
        # Check for required columns
        _validate_required_columns(data)
        
        # Check data quality
        _validate_data_quality(data)
        
        return TimeseriesDataset(data)

.. _performance-guidelines:

Performance Guidelines
======================

Pandas and NumPy optimization
-----------------------------

Follow efficient pandas patterns:

.. code-block:: python

    # Good: Vectorized operations
    def calculate_rolling_mean(data: pd.Series, window: int) -> pd.Series:
        """Calculate rolling mean using efficient pandas methods."""
        return data.rolling(window=window, min_periods=1).mean()

    # Good: Use .loc for explicit indexing
    def filter_business_hours(data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to business hours only."""
        is_business_hour = (data.index.hour >= 8) & (data.index.hour <= 18)
        return data.loc[is_business_hour]

    # Avoid: Explicit loops where vectorization is possible
    def slow_rolling_mean(data: pd.Series, window: int) -> pd.Series:
        """Inefficient implementation using loops."""
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            result.append(data.iloc[start_idx:i+1].mean())
        return pd.Series(result, index=data.index)

Memory management
-----------------

Be mindful of memory usage with large datasets:

.. code-block:: python

    def process_large_dataset(data: pd.DataFrame) -> pd.DataFrame:
        """Process large datasets efficiently."""
        
        # Use categorical data for string columns
        if 'location' in data.columns:
            data['location'] = data['location'].astype('category')
            
        # Use appropriate numeric dtypes
        data['temperature'] = data['temperature'].astype('float32')
        
        # Process in chunks if needed
        if len(data) > 1_000_000:
            return _process_in_chunks(data)
        
        return _process_all_at_once(data)

.. _backwards-compatibility:

Backwards Compatibility
=======================

API design
----------

Design stable public APIs:

.. code-block:: python

    def create_forecaster(
        model_type: str = "xgboost",
        *,
        config: Optional[ModelConfig] = None,
        **kwargs: Any
    ) -> BaseForecaster:
        """Create a forecaster with backward-compatible interface.
        
        Args:
            model_type: Type of model to create.
            config: Model configuration (recommended).
            **kwargs: Legacy parameter support (deprecated).
        """
        if kwargs and config is None:
            # Support legacy parameter format
            _log.warning(
                "Using **kwargs for configuration is deprecated. "
                "Use 'config' parameter instead."
            )
            config = ModelConfig(**kwargs)
        
        return _create_forecaster_internal(model_type, config or ModelConfig())

Deprecation warnings
--------------------

Use proper deprecation warnings for API changes:

.. code-block:: python

    import warnings
    from typing import Any

    def old_function_name(*args: Any, **kwargs: Any) -> Any:
        """Deprecated function with clear migration path."""
        warnings.warn(
            "old_function_name is deprecated and will be removed in v5.0. "
            "Use new_function_name instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return new_function_name(*args, **kwargs)

.. _getting-help-with-style:

.. include:: _getting_help.rst

Additional style resources:

* `PEP 8 Style Guide <https://peps.python.org/pep-0008/>`_
* `Ruff Documentation <https://docs.astral.sh/ruff/>`_
* `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_
* `Python Type Hints Documentation <https://docs.python.org/3/library/typing.html>`_
