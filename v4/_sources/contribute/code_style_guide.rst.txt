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
    from typing import Any

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
    from openstef.transforms.time_domain import LagTransform
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
    from openstef.transforms import LagTransform

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
    # Prefer typing with explicit dtypes using numpy.typing
    # e.g., npt.NDArray[np.float64] for floating-point arrays
    y_true: npt.NDArray[np.float64]          # Ground truth values
    y_pred: npt.NDArray[np.float64]          # Model predictions
    forecast: pd.DataFrame       # Forecast output with uncertainty
    residuals: npt.NDArray[np.float64]       # Prediction errors

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

OpenSTEF 4.0 uses type hints for better code clarity and IDE support.

Required type hints
-------------------

All **public functions** and **class methods** must have type hints:

.. code-block:: python

    def train_forecaster(
        data: TimeSeriesDataset, 
        config: ModelConfig,
        validation_split: float = 0.2
    ) -> TrainingResult:
        """Train a forecasting model with the given configuration."""

    class XGBoostForecaster:
        def __init__(self, config: XGBoostConfig) -> None:
            self.config = config

        def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "XGBoostForecaster":
            # Implementation
            return self

Common type patterns
--------------------

Use these standard patterns for common OpenSTEF types:

.. code-block:: python

    from typing import Any
    from pathlib import Path
    import numpy as np
    import numpy.typing as npt

    # File paths - prefer pathlib.Path
    def save_model(model: ForecastModel, path: Path) -> None:
        pass

    # Optional parameters with defaults
    def create_forecaster(
        model_type: str = "xgboost",
        config: ModelConfig | None = None
    ) -> BaseForecaster:
        pass

    # Union types for flexible inputs
    def load_data(source: str | Path | pd.DataFrame) -> TimeSeriesDataset:
        pass

    # Generic containers
    def validate_features(features: list[str]) -> dict[str, bool]:
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
        data: TimeSeriesDataset,
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
    def calculate_mae(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Standalone utility function."""
        return np.mean(np.abs(y_true - y_pred))

    # Level 2: Standardized components  
    class LagTransform(BaseTransform):
        """Configurable component with fit/transform API."""
        
        def fit(self, data: TimeSeriesDataset) -> "LagTransform":
            # Learn parameters if needed
            return self
            
        def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
            # Apply transformation
            pass

    # Level 3: Multi-component pipelines
    class TrainingPipeline:
        """Orchestrates multiple Level 2 components."""
        
        def __init__(
            self,
            transforms: list[BaseTransform],
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
:doc:`document` for documentation guidelines.

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

.. _logging-for-contributors:

Logging for Contributors
========================

OpenSTEF uses Python's standard logging library with a ``NullHandler`` by default, 
allowing users to configure logging as needed. Contributors should follow these 
practices for consistent and useful logging throughout the codebase.

Getting a logger
----------------

Always use ``logging.getLogger(__name__)`` to get a logger instance:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)

    def train_forecaster(data: TimeSeriesDataset) -> ForecastModel:
        """Train a forecasting model."""
        
        logger.info(f"Training model with {len(data)} samples")
        logger.debug(f"Feature columns: {data.feature_columns}")
        
        try:
            model = self._fit_model(data)
            logger.info("Model training completed successfully")
            return model
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelTrainingError(f"Training failed: {e}") from e

**Why use ``__name__``?**

The ``__name__`` variable contains the full module path (e.g., ``openstef_models.forecasting.xgboost``), 
which provides several benefits:

* **Package-level control**: Users can disable logging for entire packages like ``openstef_models``
* **Module-level granularity**: Users can control logging for specific modules like ``openstef_models.transforms``
* **Hierarchical structure**: Follows Python logging's hierarchical naming convention
* **Easy filtering**: Log output clearly shows which module generated each message

.. note::
   
   OpenSTEF also provides a more advanced logging system through the main ``openstef`` package 
   with support for structured logging. However, individual packages (``openstef-models``, 
    ``openstef-beam``, ``openstef-core``, etc.) use standard Python logging with ``NullHandler`` by default to 
   give users complete control over logging configuration. See the :doc:`../user_guide/logging` 
   guide for details on both approaches.

Logging levels and usage
------------------------

Use appropriate logging levels for different types of information:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)

    def process_data(data: pd.DataFrame) -> pd.DataFrame:
        """Process time series data."""
        
        # DEBUG: Detailed diagnostic information
        logger.debug(f"Processing data with shape {data.shape}")
        logger.debug(f"Data columns: {list(data.columns)}")
        logger.debug(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # INFO: Important operational information
        logger.info(f"Starting data processing for {len(data)} samples")
        
        # Check for potential issues
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            # WARNING: Something unexpected but recoverable
            logger.warning(f"Found {missing_count} missing values, will interpolate")
            data = data.interpolate()
        
        try:
            processed_data = _apply_transformations(data)
            logger.info("Data processing completed successfully")
            return processed_data
        except ValueError as e:
            # ERROR: Serious problem that prevents completion
            logger.error(f"Data processing failed: {e}")
            raise DataProcessingError(f"Processing failed: {e}") from e

**Logging level guidelines:**

* **DEBUG**: Detailed diagnostic information (data shapes, column names, internal states)
* **INFO**: Important operational milestones (processing started/completed, model trained)
* **WARNING**: Unexpected situations that don't prevent execution (missing data, deprecated usage)
* **ERROR**: Serious problems that prevent a function from completing
* **CRITICAL**: Very serious errors that may crash the application (rarely used)

Adding context with extras
--------------------------

Enhance log messages with structured context using the ``extra`` parameter:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)

    def train_model(data: TimeSeriesDataset, config: ModelConfig) -> TrainedModel:
        """Train a forecasting model with detailed logging context."""
        
        # Add extra context to log messages
        training_context = {
            "model_type": config.model_type,
            "training_samples": len(data),
            "feature_count": len(data.feature_columns),
            "horizon": config.horizon,
            "validation_split": config.validation_split
        }
        
        logger.info(
            f"Starting model training with {config.model_type}",
            extra={"training_info": training_context}
        )
        
        # Log performance metrics with context
        start_time = time.time()
        model = _fit_model(data, config)
        training_time = time.time() - start_time
        
        performance_context = {
            "training_time_seconds": round(training_time, 2),
            "model_size_mb": round(_get_model_size(model) / 1024 / 1024, 2),
            "convergence_iterations": getattr(model, 'n_estimators_', None)
        }
        
        logger.info(
            "Model training completed",
            extra={"performance_info": performance_context}
        )
        
        return model

**Benefits of using extras:**

* **Structured logging**: Context can be extracted by log processors
* **Better filtering**: Users can filter logs based on model type, data size, etc.
* **Monitoring integration**: Production systems can alert on specific conditions
* **Debugging**: Rich context helps identify issues without code changes

Using LoggerAdapter for consistent context
------------------------------------------

When you need to use the same context across multiple log messages or throughout 
a function/class, use ``logging.LoggerAdapter``:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)

    class FeatureEngineeringPipeline:
        """Feature engineering pipeline with consistent logging context."""
        
        def __init__(self, config: FeatureConfig):
            self.config = config
            
            # Create adapter with consistent context
            self.logger = logging.LoggerAdapter(
                logger=logger,
                extra={
                    "pipeline_id": config.pipeline_id,
                    "feature_types": config.feature_types,
                    "transform_count": len(config.transforms)
                }
            )
        
        def fit_transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
            """Apply feature engineering transformations."""
            
            # All log messages will include the adapter's extra context
            self.logger.info(f"Starting feature engineering on {len(data)} samples")
            
            result = data
            for i, transform in enumerate(self.config.transforms):
                self.logger.debug(f"Applying transform {i+1}: {transform.__class__.__name__}")
                
                try:
                    result = transform.fit_transform(result)
                    self.logger.debug(f"Transform {i+1} completed, output shape: {result.shape}")
                except Exception as e:
                    # Extra context automatically included
                    self.logger.error(f"Transform {i+1} failed: {e}")
                    raise
            
            self.logger.info(f"Feature engineering completed, output shape: {result.shape}")
            return result

    def process_multiple_datasets(datasets: list[TimeSeriesDataset]) -> list[TimeSeriesDataset]:
        """Process multiple datasets with batch context."""
        
        # Create adapter for batch processing context
        batch_logger = logging.LoggerAdapter(
            logger=logger,
            extra={
                "batch_size": len(datasets),
                "batch_id": f"batch_{int(time.time())}",
                "processing_mode": "parallel"
            }
        )
        
        batch_logger.info("Starting batch processing")
        
        results = []
        for i, dataset in enumerate(datasets):
            # Create per-dataset adapter that inherits batch context
            dataset_logger = logging.LoggerAdapter(
                logger=batch_logger,  # Chain adapters for nested context
                extra={
                    "dataset_index": i,
                    "dataset_size": len(dataset)
                }
            )
            
            dataset_logger.debug("Processing dataset")
            result = _process_single_dataset(dataset)
            dataset_logger.debug("Dataset processing completed")
            results.append(result)
        
        batch_logger.info("Batch processing completed")
        return results

**When to use LoggerAdapter:**

* **Class-level context**: When a class needs consistent context across methods
* **Function-level context**: When a function performs multiple operations with shared context
* **Nested operations**: When you need to add context at multiple levels (batch → dataset → operation)
* **Long-running processes**: When tracking context throughout extended operations

Performance considerations
---------------------------

Follow these practices to ensure logging doesn't impact performance:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)

    def performance_sensitive_function(large_data: pd.DataFrame) -> pd.DataFrame:
        """Function that processes large amounts of data efficiently."""
        
        # Good: Log level is checked before expensive operations
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processing data with columns: {list(large_data.columns)}")
            logger.debug(f"Memory usage: {large_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Good: Use lazy evaluation for expensive string formatting
        logger.info("Processing started for %d samples", len(large_data))
        
        # Avoid: Expensive operations in log arguments that are always evaluated
        # logger.debug(f"Data summary: {large_data.describe().to_string()}")  # Bad!
        
        # Good: Conditional expensive logging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Data summary: {large_data.describe().to_string()}")
        
        for chunk in _process_in_chunks(large_data):
            # Good: Minimal context for frequent operations
            logger.debug("Processed chunk of size %d", len(chunk))
        
        logger.info("Processing completed")
        return processed_data

**Performance guidelines:**

* Use ``logger.isEnabledFor(level)`` for expensive debug operations
* Prefer ``%`` formatting or ``.format()`` over f-strings in log calls for lazy evaluation
* Avoid expensive computations in log message arguments
* Use appropriate log levels to control output volume
* Consider using logging filters for fine-grained performance control

Error handling with logging
---------------------------

Combine proper exception handling with informative logging:

.. code-block:: python

    import logging

    logger = logging.getLogger(__name__)

    def robust_data_loading(file_path: Path) -> TimeSeriesDataset:
        """Load data with error handling and logging."""
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Validate file exists and is readable
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            logger.debug(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Attempt to load data
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(data)} rows, {len(data.columns)} columns")
            
            # Validate data structure
            _validate_data_structure(data)
            
            return TimeSeriesDataset(data)
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise DataLoadingError(f"Cannot load data: {e}") from e
        
        except pd.errors.EmptyDataError as e:
            logger.error(f"File is empty: {file_path}")
            raise DataLoadingError(f"Empty data file: {file_path}") from e
        
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV file: {e}")
            # Include file context in error
            logger.error(f"File path: {file_path}", extra={"file_info": {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0
            }})
            raise DataLoadingError(f"Invalid CSV format: {e}") from e
        
        except Exception as e:
            # Catch-all with full context
            logger.error(f"Unexpected error loading data: {e}", extra={
                "error_context": {
                    "file_path": str(file_path),
                    "error_type": type(e).__name__,
                    "file_exists": file_path.exists()
                }
            })
            raise DataLoadingError(f"Failed to load data from {file_path}: {e}") from e

**Error logging guidelines:**

* Always log errors before raising exceptions
* Include relevant context in error messages
* Use exception chaining (``raise ... from e``) to preserve stack traces
* Provide actionable error messages when possible
* Use structured logging for error context that might be analyzed programmatically

Testing logging behavior
------------------------

Write tests to verify logging behavior in your code:

.. code-block:: python

    import logging
    import pytest
    from unittest.mock import patch

    def test_successful_training_logs_info_messages(caplog):
        """Test that successful training produces expected log messages."""
        
        with caplog.at_level(logging.INFO):
            model = train_forecaster(sample_data, sample_config)
        
        # Verify expected log messages
        assert "Starting model training" in caplog.text
        assert "Model training completed successfully" in caplog.text
        
        # Verify log levels
        info_messages = [record for record in caplog.records if record.levelno == logging.INFO]
        assert len(info_messages) >= 2

    def test_training_failure_logs_error_with_context():
        """Test that training failures log appropriate error messages."""
        
        invalid_config = ModelConfig(learning_rate=-1.0)  # Invalid config
        
        with pytest.raises(ModelTrainingError):
            with patch('openstef_models.logger') as mock_logger:
                train_forecaster(sample_data, invalid_config)
        
        # Verify error was logged
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        assert "training failed" in error_call.lower()

    def test_logger_adapter_includes_context():
        """Test that LoggerAdapter includes expected context in log messages."""
        
        with patch('openstef_models.logger') as mock_logger:
            pipeline = FeatureEngineeringPipeline(sample_config)
            pipeline.fit_transform(sample_data)
        
        # Verify logger adapter was used with correct context
        adapter_calls = [call for call in mock_logger.info.call_args_list 
                        if 'pipeline_id' in str(call)]
        assert len(adapter_calls) > 0

**Testing guidelines:**

* Use ``caplog`` fixture to capture log messages in tests
* Test both successful operations and error conditions  
* Verify log levels are appropriate for different scenarios
* Test that structured context is included in log messages
* Mock loggers when testing complex logging logic

.. _testing-guidelines:

Testing and Code Quality
========================

Test naming and structure
-------------------------

Use concise, consistent test names following these patterns:

**For testing functions or methods:**
``test_function_name__behavior`` or ``test_function_name__condition``

**For testing classes:**
``test_class_name__method__behavior`` or ``test_class_name__behavior``

.. code-block:: python

    # Good: Concise function testing
    def test_lag_transform__creates_expected_features():
        """Test that LagTransform creates expected number of lag features."""
        pass

    def test_calculate_mae__handles_missing_values():
        """Test MAE calculation with missing data."""
        pass

    # Good: Class method testing
    def test_forecaster__fit__raises_error_empty_data():
        """Test that forecaster raises error when fitting on empty data."""
        pass

    def test_forecaster__predict__returns_forecast_dataset():
        """Test that predict returns proper ForecastDataset."""
        pass

    # Good: Class behavior testing
    def test_training_pipeline__state_serialize_restore():
        """Test pipeline state serialization and restoration."""
        pass

**Naming guidelines:**

* Use double underscores (``__``) to separate components: ``test_<subject>__<behavior>``
* Keep behavior descriptions concise but clear
* Focus on **what** is being tested, not implementation details
* Use verbs for behaviors: ``raises_error``, ``returns_value``, ``creates_features``
* Group related tests with consistent prefixes

.. code-block:: python

    # Example: Consistent test grouping for a forecaster class
    def test_constant_median_forecaster__fit_predict():
        """Test basic fit and predict workflow."""
        pass

    def test_constant_median_forecaster__predict_not_fitted_raises_error():
        """Test error when predicting without fitting."""
        pass

    def test_constant_median_forecaster__state_serialize_restore():
        """Test state persistence functionality."""
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

Use dataclasses or Pydantic models for configuration with proper validation:

.. code-block:: python

    from pydantic import BaseModel, Field, field_validator
    from typing import Literal

    class ModelConfig(BaseModel):
        """Configuration for forecasting models with automatic validation."""
        
        model_type: Literal["xgboost", "linear", "neural"] = Field(
            default="xgboost",
            description="Type of forecasting model to use"
        )
        
        learning_rate: float = Field(
            default=0.1,
            gt=0.0,
            le=1.0,
            description="Learning rate for model training"
        )
        
        max_depth: int = Field(
            default=6,
            ge=1,
            le=20,
            description="Maximum tree depth for tree-based models"
        )
        
        n_estimators: int = Field(
            default=100,
            ge=1,
            le=10000,
            description="Number of estimators/trees in ensemble"
        )
        
        random_state: int | None = Field(
            default=None,
            description="Random seed for reproducibility"
        )
        
        feature_columns: list[str] = Field(
            default_factory=list,
            description="List of feature column names to use"
        )
        
        @field_validator('learning_rate')
        @classmethod
        def validate_learning_rate(cls, v: float) -> float:
            """Ensure learning rate is reasonable for the model type."""
            if v < 0.001:
                raise ValueError("Learning rate too small, may cause slow convergence")
            return v
        
        @field_validator('feature_columns')
        @classmethod
        def validate_features(cls, v: list[str]) -> list[str]:
            """Validate feature column names."""
            if len(v) == 0:
                raise ValueError("At least one feature column must be specified")
            
            invalid_names = [name for name in v if not name.replace('_', '').isalnum()]
            if invalid_names:
                raise ValueError(f"Invalid feature names: {invalid_names}")
            
            return v

**Benefits of Pydantic configuration:**

* **Automatic validation**: Field constraints are enforced automatically
* **Type conversion**: Automatic conversion between compatible types
* **Documentation**: Field descriptions serve as inline documentation
* **Serialization**: Built-in JSON serialization for configuration persistence
* **IDE support**: Full type hints and autocompletion

Data validation patterns
------------------------

Implement data validation using clear, testable functions:

.. code-block:: python

    def validate_timeseries_data(data: pd.DataFrame) -> TimeSeriesDataset:
        """Validate and convert DataFrame to TimeSeriesDataset.
        
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
        
        return TimeSeriesDataset(data)

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
        config: ModelConfig | None = None,
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
