Changelog
=========

This page documents the version history of OpenSTEF, including new features, breaking changes, and bug fixes for each release. For step-by-step instructions on upgrading between major versions, see the :doc:`user_guide/migration` guide.

.. note::

   OpenSTEF follows `Semantic Versioning <https://semver.org/>`_. Major versions indicate breaking API changes, minor versions add backwards-compatible features, and patch versions contain bug fixes.


Version 4.0 (Current)
----------------------

OpenSTEF 4.0 is a major architectural refactor that restructures the library into a modular mono-repo with independently installable packages. This release focuses on flexibility, modularity, and broader domain applicability while maintaining the forecasting capabilities of V3. It is currently in production at Alliander with 10,000+ forecasts daily.

New Features
^^^^^^^^^^^^

**Modular Package Architecture**

The single ``openstef`` package has been split into focused, composable packages:

- ``openstef-core`` — Data types, interfaces, base classes, shared exceptions, and testing utilities
- ``openstef-models`` — Forecasting models, preprocessing pipelines, energy-specific transforms, explainability, and presets
- ``openstef-beam`` — Backtesting, Evaluation, Analysis, and Metrics
- ``openstef-meta`` — Meta-learning and modern ensemble models

Each package can be installed independently or together via the top-level ``openstef`` meta-package:

.. code-block:: bash

   # Install everything
   pip install openstef

   # Install only what you need
   pip install openstef-core
   pip install openstef-models[lgbm]
   pip install openstef-beam

**Data Availability Constraints**

V4 introduces first-class support for data availability constraints through versioned time series datasets. Transforms like ``VersionedLagsAdder`` ensure that lag features only use data that would have been available at prediction time:

.. code-block:: python

   from datetime import timedelta
   from openstef_models.transforms.time_domain.versioned_lags_adder import VersionedLagsAdder

   transform = VersionedLagsAdder(
       feature="load",
       lags=[timedelta(hours=-1), timedelta(hours=-2)],
   )
   result = transform.transform(dataset)

**Flexible Configuration**

Hard-coded assumptions have been replaced with flexible configuration mechanisms using Pydantic-based configuration classes. This enables customization of holiday calendars, data formats, and model parameters without modifying core code.

**Type Safety**

Full type safety throughout the codebase using Pydantic models and Python type annotations, catching bugs early and improving IDE support.

**Optional Model Backends**

Model backends are now optional dependencies, allowing you to install only what you need:

.. code-block:: bash

   pip install openstef-models[lgbm]      # LightGBM support
   pip install openstef-models[xgb-cpu]   # XGBoost (CPU)
   pip install openstef-models[xgb-gpu]   # XGBoost (GPU)
   pip install openstef-models[tuning]    # Optuna hyperparameter tuning

**Benchmark Datasets**

Built-in benchmark dataset support via HuggingFace Hub for reproducible evaluation:

.. code-block:: bash

   pip install openstef-core[benchmark]

**Broader Domain Support**

- Customizable holiday calendars (no longer Netherlands-specific)
- Support for diverse data formats and structures
- Applicable to district heating, dynamic pricing, and non-DSO/TSO use cases

Breaking Changes
^^^^^^^^^^^^^^^^

.. warning::

   V4 is a major release with significant API changes. See the :doc:`user_guide/migration` guide for detailed upgrade instructions.

- **Package restructuring** — All imports have changed. The monolithic ``openstef`` namespace is replaced by ``openstef_core``, ``openstef_models``, ``openstef_beam``, and ``openstef_meta``.
- **Decoupled external dependencies** — MLflow, openstef-dbc, and specific model libraries (XGBoost, LightGBM) are no longer required by default.
- **Removed hard-coded assumptions** — Netherlands-specific logic, rigid input data constraints, and fixed preprocessing pipelines have been generalized or made configurable.
- **New data types** — ``TimeSeriesDataset`` and ``VersionedTimeSeriesDataset`` replace raw DataFrames in many interfaces.
- **Configuration via Pydantic** — Dictionary-based configuration replaced with typed Pydantic models (``BaseConfig`` subclasses).
- **Centralized preprocessing** — Duplicated validation and preprocessing logic consolidated into the transforms system.

.. mermaid:: /diagrams/root/changelog_diagram_1.mmd

Improvements
^^^^^^^^^^^^

- Increased test coverage and streamlined test execution
- Standardized coding practices and documentation styles
- Comprehensive documentation following the Diátaxis framework
- Clear separation between library and reference implementation
- Improved onboarding experience with tutorials and examples
- Support for research notebooks, small-scale deployments, and enterprise integration


Version 3.x (Previous Stable)
------------------------------

Version 3.0 was the first major open-source release of OpenSTEF under the LF Energy umbrella. It established the core forecasting capabilities used in production at Alliander.

Key Features (V3)
^^^^^^^^^^^^^^^^^^

- XGBoost and LightGBM-based forecasting models
- Quantile regression for probabilistic forecasts
- Automated feature engineering (weather, temporal, lag features)
- MLflow integration for model tracking
- Database connector (openstef-dbc) for operational deployments
- Basic backtesting and evaluation pipelines

Known Limitations (V3)
^^^^^^^^^^^^^^^^^^^^^^^

These limitations motivated the V4 redesign:

- Tight coupling to MLflow and openstef-dbc made standalone use difficult
- Netherlands-specific assumptions (holidays, data formats) limited international adoption
- Rigid input data constraints required specific column names and formats
- Monolithic architecture made it difficult to extend with custom models
- Duplicated preprocessing logic across validation and model components
- Limited documentation and unclear distinction between library and reference implementation


Version History Summary
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Version
     - Status
     - Highlights
   * - 4.0
     - Current (Alpha)
     - Modular mono-repo, type safety, data availability constraints, flexible configuration, decoupled dependencies
   * - 3.x
     - Previous Stable
     - First open-source release, XGBoost/LightGBM models, MLflow integration, production-proven at Alliander


Versioning Policy
-----------------

OpenSTEF uses semantic versioning with the following guarantees:

- **Major versions** (e.g., 3.x → 4.x): May contain breaking API changes. Migration guides are provided.
- **Minor versions** (e.g., 4.0 → 4.1): New features added in a backwards-compatible manner.
- **Patch versions** (e.g., 4.0.0 → 4.0.1): Bug fixes only, no API changes.

Individual packages within the mono-repo (``openstef-core``, ``openstef-models``, ``openstef-beam``, ``openstef-meta``) are versioned together to ensure compatibility.


Getting Help with Upgrades
--------------------------

If you are upgrading from V3 to V4:

- Start with the :doc:`user_guide/migration` guide for step-by-step instructions
- Review the :doc:`user_guide/installation` page for the new package structure
- Check the :doc:`tutorials/index` for updated examples using V4 APIs
- Open an issue on `GitHub <https://github.com/OpenSTEF/openstef>`_ if you encounter problems