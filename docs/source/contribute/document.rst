.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

.. _documenting-openstef:

===================
Documentation Guide
===================

OpenSTEF welcomes improvements to documentation! Whether you're fixing typos, clarifying 
explanations, or adding comprehensive tutorials, your contributions help make forecasting 
more accessible to everyone.

Getting started
===============

Documentation structure
-----------------------

OpenSTEF documentation is organized following the `Diátaxis framework <https://diataxis.fr/>`_:

* **Tutorials** (``examples/tutorials/``): Learning-oriented guides for beginners
* **How-to guides** (``user_guide/``): Problem-oriented practical guides  
* **Reference** (``api/``): Information-oriented technical reference
* **Explanation** (``user_guide/intro/``): Understanding-oriented background material

The documentation lives in several places:

.. code-block:: text

    docs/source/          # Main documentation source
    ├── api/              # API reference (auto-generated)
    ├── user_guide/       # User guides and tutorials  
    ├── project/          # Project information
    ├── contribute/       # Contributing guides
    └── examples/         # Example gallery

    examples/             # Example scripts and tutorials
    ├── examples/         # Gallery examples
    └── tutorials/        # Comprehensive tutorials

    packages/*/src/       # Inline code documentation (docstrings)

Building the documentation
==========================

To build the documentation locally:

.. code-block:: bash

    # Build the documentation
    poe docs

    # Build and serve with live reload (recommended for editing)
    poe docs --serve

    # Clean previous builds
    poe docs-clean

The built documentation will be available at ``docs/build/html/index.html``.

.. note::

    Building documentation requires additional dependencies that are included in the 
    development environment. Make sure you've run ``uv sync --group dev`` first.

Writing docstrings
==================

OpenSTEF uses `Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ 
for all code documentation. This style is clear, readable, and well-supported by Sphinx.

Basic docstring structure
-------------------------

.. code-block:: python

    def forecast_energy(data: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Generate energy forecasts for the specified horizon.

        This function creates forecasts using the configured model and feature
        engineering pipeline. It handles missing data and provides uncertainty
        estimates for each prediction.

        Args:
            data: Historical energy consumption data with datetime index.
                Must include columns: ['load', 'temperature', 'wind_speed'].
            horizon: Number of hours to forecast ahead. Must be positive.

        Returns:
            DataFrame with forecasted values and uncertainty bounds:
                - 'forecast': Point predictions
                - 'forecast_lower': Lower confidence bound (5th percentile)  
                - 'forecast_upper': Upper confidence bound (95th percentile)

        Raises:
            ValueError: If data is empty or missing required columns.
            TypeError: If horizon is not a positive integer.

        Example:
            Basic usage with sample data:

            >>> import pandas as pd
            >>> data = pd.DataFrame({
            ...     'load': [100, 120, 110],
            ...     'temperature': [20, 22, 21],
            ...     'wind_speed': [5, 7, 6]
            ... }, index=pd.date_range('2025-01-01', periods=3, freq='h'))
            >>> forecast = forecast_energy(data, horizon=6)
            >>> forecast.shape
            (6, 3)

        Note:
            The model automatically handles daylight saving time transitions and
            public holidays for improved accuracy.

        See Also:
            evaluate_forecast: Evaluate forecast accuracy against ground truth.
            prepare_features: Prepare input data for forecasting.
        """

Docstring sections
------------------

Use these sections in your docstrings (order matters):

1. **Summary line**: One-line description of what the function does
2. **Extended description**: More detailed explanation (optional)
3. **Args**: Function parameters and their types/descriptions
4. **Returns**: Description of return value(s)
5. **Raises**: Exceptions that may be raised
6. **Example**: Code examples showing how to use the function
7. **Note**: Additional important information
8. **Invariants**: Contract guarantees and requirements (for classes and interfaces)
9. **See Also**: References to related functions/classes

Type hints and docstrings
-------------------------

Always use type hints in function signatures. The docstring should complement, not repeat, 
the type information:

.. code-block:: python

    # Good: Type hints in signature, description in docstring
    def train_model(data: TimeseriesDataset, config: ModelConfig) -> ForecastModel:
        """Train a forecasting model on the provided dataset.
        
        Args:
            data: Training dataset with features and targets.
            config: Model configuration including hyperparameters.
            
        Returns:
            Trained model ready for forecasting.
        """

    # Avoid: Repeating type information in docstring
    def train_model(data: TimeseriesDataset, config: ModelConfig) -> ForecastModel:
        """Train a forecasting model on the provided dataset.
        
        Args:
            data (TimeseriesDataset): Training dataset with features and targets.
            config (ModelConfig): Model configuration including hyperparameters.
            
        Returns:
            ForecastModel: Trained model ready for forecasting.
        """

Invariants section
------------------

For classes and interfaces, use an ``Invariants`` section to document the contract 
guarantees and requirements that implementers and users must follow:

.. code-block:: python

    class ForecastModel(ABC):
        """Base class for all forecasting models.

        Provides a standardized interface for training and prediction across
        different forecasting algorithms and approaches.

        Invariants:
            - fit() must be called before predict() for stateful models
            - predict() should handle all horizons specified in configuration
            - Output format must be consistent with ForecastDataset structure
            - Model state must remain unchanged during prediction calls

        Example:
            Basic model implementation:

            >>> class SimpleModel(ForecastModel):
            ...     def fit(self, data):
            ...         self._fitted = True
            ...     def predict(self, data):
            ...         return generate_forecasts(data)
        """

The ``Invariants`` section should document:

* **Pre-conditions**: What must be true before calling methods
* **Post-conditions**: What the implementation guarantees after execution  
* **State requirements**: How object state should be managed
* **Interface contracts**: Consistent behavior expectations across implementations

This helps both implementers understand what they must provide and users understand 
what they can rely on.

.. note::

    The ``Invariants`` section is an OpenSTEF-specific extension to Google-style 
    docstrings. Use it for classes and interfaces where contract guarantees are 
    important for correct implementation and usage.

Examples in docstrings
======================

Include practical examples in your docstrings using the ``Example`` section. 
These examples are automatically tested with ``poe doctests``.

Writing good examples
---------------------

.. code-block:: python

    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error between predictions and ground truth.

        Example:
            Basic usage:

            >>> import numpy as np
            >>> y_true = np.array([1, 2, 3, 4, 5])
            >>> y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
            >>> mae = calculate_mae(y_true, y_pred)
            >>> round(mae, 2)
            0.16

            With perfect predictions:

            >>> perfect_pred = np.array([1, 2, 3, 4, 5])
            >>> calculate_mae(y_true, perfect_pred)
            0.0
        """

Example guidelines
------------------

* **Keep examples simple** but realistic
* **Use ``>>>`` prompts** for interactive examples
* **Show expected output** when it's not obvious
* **Test edge cases** (empty inputs, perfect predictions, etc.)
* **Use ``round()``** for floating-point outputs to avoid precision issues
* **Import required modules** within the example if needed

Writing narrative documentation
===============================

For user guides, tutorials, and explanatory content, use reStructuredText (.rst) files.

reStructuredText basics
-----------------------

.. code-block:: rst

    Section headers
    ===============

    Subsection headers
    ------------------

    *Italic text* and **bold text**

    ``Code snippets`` and :func:`function references`

    .. code-block:: python

        # Code blocks with syntax highlighting
        import openstef
        model = openstef.create_model()

    .. note::

        Informational notes for readers.

    .. warning::

        Important warnings about potential issues.

Cross-references
----------------

Link to other parts of the documentation:

.. code-block:: rst

    # Link to other documents
    See the :doc:`user_guide/installation` guide.

    # Link to specific functions/classes  
    Use :func:`openstef.models.forecast` for predictions.
    
    # Link to sections within documents
    Refer to :ref:`development_setup` for setup instructions.

Contributing to examples and tutorials
======================================

Examples and tutorials are crucial for user onboarding. When adding new examples:

1. **Choose the right location**:
   
   * ``examples/examples/`` - Short, focused examples
   * ``examples/tutorials/`` - Comprehensive, multi-step tutorials

2. **Follow naming conventions**:
   
   * Use descriptive filenames: ``basic_forecasting.py``, ``advanced_transforms.py``
   * Start with a docstring explaining the example's purpose

3. **Structure your example**:

   .. code-block:: python

       """
       Basic Energy Forecasting
       ========================
       
       This example demonstrates how to create simple energy forecasts using OpenSTEF.
       We'll load sample data, train a model, and generate predictions.
       """

       import pandas as pd
       import openstef

       # Load sample data
       data = openstef.load_sample_data()
       
       # ... rest of example

4. **Include explanations**: Use comments and markdown cells to explain each step

5. **Test your examples**: Run ``poe doctests`` to ensure all examples work

Documentation style guide
=========================

Writing style
-------------

* **Be clear and concise** - Avoid jargon, explain technical terms
* **Use active voice** - "The model predicts" rather than "Predictions are made"
* **Write for your audience** - Tutorials for beginners, reference for experts
* **Include context** - Explain why something is useful, not just how to do it

Code style in documentation
---------------------------

* **Use realistic examples** - Avoid ``foo``, ``bar``; use domain-relevant names
* **Show complete examples** - Include imports and setup code
* **Highlight important parts** - Use comments to draw attention to key concepts
* **Test all code** - Ensure examples actually work

All code examples in documentation should follow our :doc:`code_style_guide`.

Building and testing documentation
==================================

Before submitting documentation changes:

.. code-block:: bash

    # Check that documentation builds without errors
    poe docs

    # Test all code examples in docstrings
    poe doctests

    # Run full quality checks (includes documentation)
    poe all --check

Common issues
-------------

* **Import errors**: Make sure all imports in examples are available
* **Outdated examples**: Keep examples current with API changes  
* **Broken links**: Verify that all cross-references work
* **Missing docstrings**: All public functions need documentation

.. include:: _getting_help.rst

Additional documentation resources
==================================

If you need help with documentation specifically:

* Check the `Sphinx documentation <https://www.sphinx-doc.org/>`_
* Look at existing documentation for examples
* Reference the `Diátaxis framework <https://diataxis.fr/>`_ for guidance on documentation types