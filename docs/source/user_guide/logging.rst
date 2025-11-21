.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

.. _logging:

=======
Logging
=======

OpenSTEF uses Python's standard logging library to provide information about its operations. 
By default, OpenSTEF configures a null handler, giving you complete control over how logging 
is handled in your application.

.. _logging-overview:

Overview
========

OpenSTEF follows these logging principles:

* **No default output**: OpenSTEF uses ``NullHandler`` by default, so no log messages appear unless you configure logging
* **Standard library**: Uses Python's built-in ``logging`` module for consistency
* **Hierarchical loggers**: Package-level and module-level loggers allow granular control
* **Structured context**: Log messages include contextual information through extras

.. _logging-configuration:

Basic Configuration
===================

To see OpenSTEF log messages, configure logging in your application:

.. code-block:: python

    import logging
    
    # Basic configuration - shows INFO level and above
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Now OpenSTEF operations will produce log output
    # TODO: Update with actual OpenSTEF classes when implemented
    from openstef_models import LinearForecaster
    forecaster = LinearForecaster()  # Will log initialization details

.. _logging-levels:

Log Levels
==========

OpenSTEF uses standard Python logging levels:

* **DEBUG**: Detailed diagnostic information, typically only of interest when diagnosing problems
* **INFO**: General information about program execution
* **WARNING**: Something unexpected happened, but the software is still working
* **ERROR**: A serious problem occurred that prevented a function from completing
* **CRITICAL**: A very serious error occurred that may prevent the program from continuing

Example configuration for different use cases:

.. code-block:: python

    import logging
    
    # Development: See everything
    logging.basicConfig(level=logging.DEBUG)
    
    # Production: Important messages only
    logging.basicConfig(level=logging.WARNING)
    
    # Data science workflows: Informational messages
    logging.basicConfig(level=logging.INFO)

.. _logger-hierarchy:

Logger Hierarchy
================

OpenSTEF loggers follow Python's hierarchical naming convention. You can control 
logging at different levels of granularity:

Package level control
---------------------

Control logging for entire OpenSTEF packages:

.. code-block:: python

    import logging
    
    # Disable all openstef-models logging
    logging.getLogger('openstef_models').setLevel(logging.CRITICAL)
    
    # Show only warnings from openstef-beam
    logging.getLogger('openstef_beam').setLevel(logging.WARNING)
    
    # Enable debug mode for specific package
    logging.getLogger('openstef_models').setLevel(logging.DEBUG)

Module level control
--------------------

Control logging for specific modules:

.. code-block:: python

    import logging

    # Only show errors from the presets module
    logging.getLogger('openstef_models.presets').setLevel(logging.ERROR)
    
    # Debug feature engineering specifically
    logging.getLogger('openstef_models.transforms').setLevel(logging.DEBUG)

.. _advanced-configuration:

Advanced Configuration
======================

For more advanced logging configurations like custom formatters, file handlers, 
rotating logs, and other features, refer to the official 
`Python logging documentation <https://docs.python.org/3/library/logging.html>`_.

.. _contextual-information:

Contextual Information
======================

OpenSTEF includes contextual information in log messages to help with debugging 
and monitoring. Many log messages include extra fields that provide additional context:

.. code-block:: python

    # Example of structured log output (when using appropriate formatters)
    2025-01-20 14:30:25 - openstef_models.training - INFO - Model training started
        extra_info: {
            'model_type': 'XGBoostForecaster',
            'training_samples': 8760,
            'features': ['temperature', 'humidity', 'hour_of_day'],
            'horizon': 24
        }

This contextual information is particularly useful when:

* Debugging model training issues
* Monitoring model performance in production
* Analyzing feature engineering pipelines
* Tracking data processing workflows

.. _integration-examples:

Integration Examples
====================

Jupyter Notebooks
------------------

Configure logging for interactive data science work:

.. code-block:: python

    import logging
    
    # Configure for notebook use
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)-20s | %(levelname)-8s | %(message)s',
        force=True  # Override any existing configuration
    )
    
    # Now OpenSTEF operations will show progress
    # TODO: Update with actual OpenSTEF classes when implemented
    from openstef_models import load_sample_data, XGBoostForecaster
    
    data = load_sample_data()  # Shows data loading progress
    model = XGBoostForecaster()
    model.fit(data)  # Shows training progress

Structured logging with structlog
---------------------------------

If you're using `structlog <https://www.structlog.org/>`_ in your application, 
you can configure OpenSTEF to work with it by integrating structlog with Python's 
standard logging:

.. code-block:: python

    import logging
    import structlog
    
    # Configure structlog to integrate with standard logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )
    
    # Now OpenSTEF logs will be processed by structlog
    # TODO: Update with actual OpenSTEF classes when implemented
    from openstef_models import create_forecaster
    forecaster = create_forecaster()

For more advanced structlog configurations and features, see the 
`structlog standard library integration guide <https://www.structlog.org/en/stable/standard-library.html>`_.

Standard logging setup
----------------------

For most applications, a simple standard logging configuration is sufficient:

.. code-block:: python

    import logging
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Optional: Adjust OpenSTEF package levels
    logging.getLogger('openstef_models').setLevel(logging.INFO)
    logging.getLogger('openstef_beam').setLevel(logging.WARNING)
    
    # Now use OpenSTEF with logging
    # TODO: Update with actual OpenSTEF classes when implemented
    from openstef_models import LinearForecaster
    forecaster = LinearForecaster()  # Will log initialization details

.. _troubleshooting:

Troubleshooting
===============

No log output appearing
-----------------------

If you're not seeing any OpenSTEF log messages:

1. **Check if logging is configured**: OpenSTEF uses ``NullHandler`` by default
2. **Verify log levels**: Ensure your handler level isn't too restrictive
3. **Check logger hierarchy**: Parent logger settings can override child settings

.. code-block:: python

    import logging
    
    # Debug logging configuration
    logger = logging.getLogger('openstef_models')
    print(f"Logger level: {logger.level}")
    print(f"Effective level: {logger.getEffectiveLevel()}")
    print(f"Handlers: {logger.handlers}")
    print(f"Parent handlers: {logger.parent.handlers}")

Too much log output
-------------------

If OpenSTEF is producing too many log messages:

.. code-block:: python

    import logging
    
    # Reduce OpenSTEF verbosity
    logging.getLogger('openstef_models').setLevel(logging.WARNING)
    logging.getLogger('openstef_beam').setLevel(logging.WARNING)
    
    # Or disable specific noisy modules
    logging.getLogger('openstef_models.transforms').setLevel(logging.ERROR)

Performance considerations
--------------------------

Logging can impact performance in tight loops. OpenSTEF follows best practices:

* Log messages are not formatted unless actually output
* Debug logging is conditionally executed
* Structured logging uses lazy evaluation

You can further optimize by:

.. code-block:: python

    import logging
    
    # Set appropriate levels to avoid unnecessary processing
    logging.getLogger('openstef_models').setLevel(logging.WARNING)
    
    # Use logging filters for fine-grained control
    class PerformanceFilter(logging.Filter):
        def filter(self, record):
            # Skip debug messages during performance-critical sections
            return record.levelno >= logging.INFO
    
    logging.getLogger('openstef_models').addFilter(PerformanceFilter())

.. _best-practices:

Best Practices
==============

For library users:

1. **Configure logging early**: Set up logging before importing OpenSTEF modules
2. **Use appropriate levels**: INFO for general monitoring, DEBUG for troubleshooting
3. **Leverage hierarchical control**: Use package/module-level logger configuration
4. **Integrate with your existing setup**: OpenSTEF works with any Python logging configuration

Example complete setup:

.. code-block:: python

    import logging
    
    def setup_openstef_logging(level=logging.INFO):
        """Set up logging for OpenSTEF integration."""
        
        # Basic configuration
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Optional: Fine-tune specific packages
        logging.getLogger('openstef_models').setLevel(level)
        logging.getLogger('openstef_beam').setLevel(logging.WARNING)  # Less verbose
    
    # Use in your application
    setup_openstef_logging(level=logging.INFO)
    
    # Now use OpenSTEF with proper logging
    # TODO: Update with actual OpenSTEF classes when implemented
    from openstef_models import create_forecaster
    forecaster = create_forecaster()
