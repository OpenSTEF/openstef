.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _installation:

=============
Installation
=============

OpenSTEF 4.0 is designed with a modular architecture that allows you to install only the components you need. The library consists of several packages that can be installed independently or together.

System Requirements
===================

* Python 3.12 or higher (Python 3.13 supported)
* 64-bit operating system (Windows, macOS, or Linux)

.. note::
   OpenSTEF 4.0 requires Python 3.12+ for optimal performance and modern type safety features. 
   If you need Python 3.10/3.11 support, consider using OpenSTEF 3.x.

Package Overview
================

OpenSTEF 4.0 follows a modular design with specialized packages:

.. list-table:: OpenSTEF Packages
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Description
   * - ``openstef``
     - Meta-package that installs the core components
   * - ``openstef-core``
     - Core utilities, dataset types, shared types and base models
   * - ``openstef-models``
     - Core ML models, feature engineering, and data processing
   * - ``openstef-beam``
     - Backtesting, Evaluation, Analysis, and Metrics (BEAM)
   * - ``openstef-meta``
     - Meta-models for combining and stacking forecasts (ensembles, weighted blends)
   * - ``openstef-foundation-models``
     - Foundation-model forecasters (e.g. Chronos-2) run on an ONNX runtime

Quick Installation
==================

For most users, start with the meta-package:

.. tab-set::
    :class: sd-width-content-min

    .. tab-item:: pip

        .. code-block:: bash

            pip install openstef

    .. tab-item:: uv

        .. code-block:: bash

            uv add openstef

    .. tab-item:: conda

        .. code-block:: bash

            conda install -c conda-forge openstef

    .. tab-item:: pixi

        .. code-block:: bash

            pixi add openstef

This installs the ``openstef`` meta-package, a minimal-but-runnable convenience layer:
``openstef-core`` plus ``openstef-models`` with its CPU XGBoost runtime. To pick GPU
runtimes, foundation models, or a leaner footprint, install the individual component
packages with the extras you need (see below).

Installation Options
====================

Choose Your Installation
-------------------------

OpenSTEF's modular design allows you to install exactly what you need:

**Complete Installation (Recommended for most users):**

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            pip install "openstef[all]"

    .. tab-item:: uv

        .. code-block:: bash

            uv add "openstef[all]"

This installs every component (``openstef-beam``, ``openstef-foundation-models``,
``openstef-meta``, ``openstef-models``) in its CPU flavour.

**Individual Package Installation:**

Install only the packages you need:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            # Core utilities and datasets only
            pip install openstef-core

            # Core forecasting models only
            pip install openstef-models
            
            # Backtesting and evaluation tools only
            pip install openstef-beam
            
            # Meta-package with models (default)
            pip install openstef

    .. tab-item:: uv

        .. code-block:: bash

            # Core utilities and datasets only
            uv add openstef-core

            # Core forecasting models only
            uv add openstef-models
            
            # Backtesting and evaluation tools only
            uv add openstef-beam
            
            # Meta-package with models (default)
            uv add openstef

**Selective Installation with Extras:**

Mix and match components using the meta-package:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            # Models + BEAM
            pip install "openstef[beam]"

            # Models + foundation models (CPU runtime)
            pip install "openstef[foundation-models]"

            # Multiple extras
            pip install "openstef[beam,foundation-models]"

    .. tab-item:: uv

        .. code-block:: bash

            # Models + BEAM
            uv add "openstef[beam]"

            # Models + foundation models (CPU runtime)
            uv add "openstef[foundation-models]"

            # Multiple extras
            uv add "openstef[beam,foundation-models]"

**Use Case Examples:**

.. list-table:: Installation by Use Case
   :header-rows: 1
   :widths: 30 40 30

   * - Use Case
     - Installation Command
     - What You Get
   * - Research & Experimentation
     - ``pip install "openstef[all]"``
     - Full toolkit for analysis
   * - Production Forecasting
     - ``pip install openstef-models``
     - Lightweight core models
   * - Model Evaluation
     - ``pip install "openstef[beam]"``
     - Models + evaluation tools
   * - Basic Development
     - ``pip install openstef``
     - Core functionality

Compute Runtimes: CPU vs GPU
----------------------------

Some packages ship a heavy compute runtime that comes in mutually exclusive CPU and
GPU builds. Pick exactly one per package: they are declared as conflicting extras, so
a resolver refuses to install both at once.

.. list-table:: Choose one runtime per package
   :header-rows: 1
   :widths: 34 33 33

   * - Package
     - CPU (default)
     - GPU (CUDA; Linux/Windows)
   * - ``openstef-foundation-models``
     - ``[cpu]`` (onnxruntime)
     - ``[gpu]`` (onnxruntime-gpu)
   * - ``openstef-models``
     - ``[xgb-cpu]``
     - ``[xgb-gpu]``

.. code-block:: bash

    # CPU build (works on every platform; the flavour the meta-package ships)
    pip install "openstef-foundation-models[cpu]"

    # GPU build (CUDA-enabled Linux or Windows only)
    pip install "openstef-foundation-models[gpu]"

The ``openstef`` meta-package (and its ``[all]`` extra) always selects the CPU builds.
For GPU, install the component package directly with its ``[gpu]`` extra. GPU wheels are
published for Linux and Windows with CUDA; there is no GPU build for macOS.

Feature extras are additive — combine as many as you need:

.. list-table:: Optional feature extras
   :header-rows: 1
   :widths: 40 60

   * - Extra
     - Adds
   * - ``openstef-models[lgbm]``
     - LightGBM forecasters
   * - ``openstef-models[tuning]``
     - Optuna hyperparameter tuning
   * - ``openstef-core[benchmark]``
     - Benchmark dataset loaders (HuggingFace Hub)
   * - ``openstef-beam[all]``
     - All BEAM baselines plus S3 storage

Development Installation
========================

For contributors and advanced users who want to modify the source code:

Prerequisites
-------------

* `uv <https://docs.astral.sh/uv/>`_ (recommended) or pip
* Git

Clone and Install
-----------------

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/OpenSTEF/openstef.git
    cd openstef

    # Install the full development environment (CPU flavour)
    uv sync

    # Verify installation
    uv run poe all

A plain ``uv sync`` installs the default ``dev`` group: every workspace package in
editable mode plus the full toolbelt (test, lint, type-check, notebooks, docs). One
command, no ``--all-groups`` or ``--all-packages`` needed.

For a GPU development environment (CUDA; Linux/Windows), swap the runtime flavour:

.. code-block:: bash

    uv sync --no-default-groups --group dev-gpu

Partial Toolbelts
-----------------

The ``dev`` group aggregates focused groups you can sync on their own, e.g. to run
just the tests or just the linters:

.. code-block:: bash

    uv sync --no-default-groups --group test    # pytest stack only
    uv sync --no-default-groups --group lint     # ruff / reuse / pyproject-fmt

.. note::

    Do not pass ``--all-groups`` or ``--all-extras``: the CPU and GPU runtimes are
    declared as conflicting extras, so activating both flavours at once fails.

Verification
============

Verify your installation:

.. code-block:: python

    import openstef_models
    print(f"OpenSTEF Models version: {openstef_models.__version__}")

    # If you installed openstef-beam
    try:
        import openstef_beam
        print(f"OpenSTEF BEAM version: {openstef_beam.__version__}")
    except ImportError:
        print("OpenSTEF BEAM not installed")

Troubleshooting
===============

Common Issues
-------------

**Python Version Error**

If you see a Python version error:

.. code-block:: text

    ERROR: Package 'openstef' requires a different Python: 3.11.0 not in '>=3.12,<4.0'

Upgrade to Python 3.12 or higher. We recommend using `pyenv <https://github.com/pyenv/pyenv>`_ or `conda <https://conda.io/>`_ to manage Python versions.

**Package Not Found**

If conda cannot find the package:

.. code-block:: bash

    # Add conda-forge channel
    conda config --add channels conda-forge
    conda install openstef

**Import Errors**

If you encounter import errors, ensure you're using the correct package names:

.. code-block:: python

    # Correct imports
    from openstef_models.presets import ForecastingWorkflowConfig
    from openstef_beam.evaluation import EvaluationPipeline

    # Not: from openstef.models import ...

**Memory Issues**

For large datasets, consider:

* Installing packages with specific memory optimizations
* Using data streaming approaches
* Configuring appropriate chunk sizes

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/OpenSTEF/openstef/issues>`_
2. Review the :doc:`/contribute/index` guide
3. Visit our :ref:`support` page for community resources
4. Contact us at openstef@lfenergy.org

Platform-Specific Notes
========================

Windows
-------

* Use PowerShell or Command Prompt
* Consider using `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/>`_ for best compatibility
* Some scientific packages may require Microsoft Visual C++ Build Tools

macOS
-----

* Most installations work out of the box
* For Apple Silicon, if you encounter errors related to OpenMP or XGBoost, install the OpenMP library via Homebrew: ``brew install libomp``

Linux
-----

* Most distributions work out of the box
* For Ubuntu/Debian: ``sudo apt-get install python3-dev``
* For RHEL/CentOS: ``sudo yum install python3-devel``

Next Steps
==========

.. seealso::

   - :doc:`quick_start_tutorial` to get started with your first forecast.
   - :doc:`/examples` for hands-on examples.
   - :doc:`/api/index` for detailed API documentation.
   - :doc:`/user_guide/concepts/index` to understand OpenSTEF's capabilities.

Staying Updated
===============

OpenSTEF follows semantic versioning. To stay updated with the latest releases:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            # Check current version
            pip show openstef

            # Upgrade to latest version
            pip install --upgrade openstef

    .. tab-item:: uv

        .. code-block:: bash

            # Check current version
            uv list | grep openstef

            # Upgrade to latest version
            uv upgrade openstef

    .. tab-item:: conda

        .. code-block:: bash

            # Check current version
            conda list openstef

            # Upgrade to latest version
            conda update openstef

    .. tab-item:: pixi

        .. code-block:: bash

            # Check current version
            pixi list | grep openstef

            # Upgrade to latest version
            pixi upgrade openstef

Subscribe to our `GitHub releases <https://github.com/OpenSTEF/openstef/releases>`_ for notifications about new versions and features.
