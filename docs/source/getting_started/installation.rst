Installation
============

This page covers how to install OpenSTEF, configure optional dependencies for your use case, verify your installation, and troubleshoot common issues.

System Requirements
-------------------

- **Python**: >=3.12, <4.0
- **Operating System**: Linux, macOS, or Windows
- **Package Manager**: pip

Quick Install
-------------

Install the complete OpenSTEF framework with all packages:

.. code-block:: bash

   pip install openstef

This meta-package installs all core components:

- ``openstef-core`` — Core data structures, datasets, and transform pipelines
- ``openstef-beam`` — Backtesting, Evaluation, Analysis and Metrics
- ``openstef-models`` — Forecasting models (LightGBM, XGBoost, etc.)
- ``openstef-meta`` — Meta-models that combine multiple forecasters

We recommend installing in a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows

   pip install openstef

Installing Individual Packages
------------------------------

If you only need specific functionality, install packages individually to keep your environment lean:

.. code-block:: bash

   # Core data structures and pipelines only
   pip install openstef-core

   # Backtesting and evaluation tools
   pip install openstef-beam

   # Forecasting models
   pip install openstef-models

   # Meta-models (installs beam, core, and models as dependencies)
   pip install openstef-meta

Optional Dependencies
---------------------

Several packages offer optional extras for specialized use cases.

openstef-models extras
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # LightGBM support
   pip install openstef-models[lgbm]

   # XGBoost on CPU (recommended for most users)
   pip install openstef-models[xgb-cpu]

   # XGBoost with GPU acceleration
   pip install openstef-models[xgb-gpu]

   # Hyperparameter tuning with Optuna
   pip install openstef-models[tuning]

openstef-beam extras
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # All optional features including baselines and S3 support
   pip install openstef-beam[all]

   # Baseline models for comparison (installs openstef-meta and openstef-models)
   pip install openstef-beam[baselines]

openstef-core extras
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Benchmark datasets from HuggingFace Hub
   pip install openstef-core[benchmark]

Combining extras
^^^^^^^^^^^^^^^^

You can combine multiple extras in a single install command:

.. code-block:: bash

   pip install openstef-models[lgbm,xgb-cpu,tuning]

Verifying Your Installation
---------------------------

After installation, verify that everything is working correctly:

.. code-block:: python

   import openstef_beam
   print(openstef_beam.__version__)

   import openstef_core
   print(openstef_core.__version__)

   import openstef_models
   print(openstef_models.__version__)

For a more thorough check, confirm that key components can be imported:

.. code-block:: python

   from openstef_core.datasets import TimeSeriesDataset, ForecastDataset
   from openstef_models.models.forecasting.forecaster import Forecaster
   from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

   print("All core imports successful!")

If you installed with LightGBM or XGBoost support, verify those as well:

.. code-block:: python

   # If you installed openstef-models[lgbm]
   import lightgbm
   print(f"LightGBM version: {lightgbm.__version__}")

   # If you installed openstef-models[xgb-cpu] or openstef-models[xgb-gpu]
   import xgboost
   print(f"XGBoost version: {xgboost.__version__}")

Troubleshooting
---------------

Python version mismatch
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   ERROR: Package requires Python >=3.12,<4.0

OpenSTEF requires Python 3.12 or later. Check your version:

.. code-block:: bash

   python --version

If you have an older version, install Python 3.12+ from `python.org <https://www.python.org/downloads/>`_ or use a version manager like ``pyenv``:

.. code-block:: bash

   pyenv install 3.12
   pyenv local 3.12

XGBoost platform issues
^^^^^^^^^^^^^^^^^^^^^^^^

On Linux and Windows, the ``xgb-cpu`` extra installs ``xgboost-cpu`` (a CPU-only build). On macOS, it installs the standard ``xgboost`` package. If you encounter build errors:

.. code-block:: bash

   # Ensure you have up-to-date pip and wheel
   pip install --upgrade pip wheel setuptools

   # Then retry
   pip install openstef-models[xgb-cpu]

LightGBM compilation errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On some systems, LightGBM requires additional system libraries:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install libgomp1

   # macOS (if using Homebrew)
   brew install libomp

Then retry:

.. code-block:: bash

   pip install openstef-models[lgbm]

Dependency conflicts
^^^^^^^^^^^^^^^^^^^^

If you encounter dependency resolution conflicts, try installing in a fresh virtual environment:

.. code-block:: bash

   python -m venv .venv-fresh
   source .venv-fresh/bin/activate
   pip install --upgrade pip
   pip install openstef

For persistent issues, install packages one at a time to identify the conflict:

.. code-block:: bash

   pip install openstef-core
   pip install openstef-models[lgbm]
   pip install openstef-beam
   pip install openstef-meta

Import errors after installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If imports fail with ``ModuleNotFoundError``, ensure you are using the correct Python environment:

.. code-block:: bash

   # Check which Python is active
   which python
   pip list | grep openstef

.. warning::

   Do not name your own scripts or directories ``openstef_core``, ``openstef_beam``, ``openstef_models``, or ``openstef_meta``. This will shadow the installed packages and cause import failures.

Development Installation
------------------------

To install OpenSTEF for development or contribution, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/OpenSTEF/openstef.git
   cd openstef
   pip install -e .

For individual packages in development mode:

.. code-block:: bash

   pip install -e "./packages/openstef-core"
   pip install -e "./packages/openstef-models[lgbm,xgb-cpu,tuning]"
   pip install -e "./packages/openstef-beam[all]"

Next Steps
----------

Once your installation is verified, proceed to:

- :doc:`quickstart` — Get a forecast running in under 5 minutes
- :doc:`first_forecast` — Detailed walkthrough of your first forecast
- :doc:`advanced_customization` — Configure custom models and pipelines