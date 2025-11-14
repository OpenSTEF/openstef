.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
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
   * - ``openstef-models``
     - Core ML models, feature engineering, and data processing
   * - ``openstef-beam``
     - Backtesting, Evaluation, Analysis, and Metrics (BEAM)
   * - ``openstef-compatibility``
     - Compatibility layer for OpenSTEF 3.x (coming soon)
   * - ``openstef-foundational-models``
     - Deep learning and foundational models (coming soon)

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

This installs ``openstef-models`` by default, which provides the core forecasting functionality.

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

This installs all available packages: ``openstef-models`` and ``openstef-beam``.

**Individual Package Installation:**

Install only the packages you need:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            # Core forecasting models only
            pip install openstef-models
            
            # Backtesting and evaluation tools only
            pip install openstef-beam
            
            # Meta-package with models (default)
            pip install openstef

    .. tab-item:: uv

        .. code-block:: bash

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
            
            # Models + Foundational models (when available)
            pip install "openstef[foundational-models]"
            
            # Multiple extras
            pip install "openstef[beam,foundational-models]"

    .. tab-item:: uv

        .. code-block:: bash

            # Models + BEAM
            uv add "openstef[beam]"
            
            # Models + Foundational models (when available)
            uv add "openstef[foundational-models]"
            
            # Multiple extras
            uv add "openstef[beam,foundational-models]"

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

    # Install in development mode with all dependencies
    uv sync --all-extras --dev

    # Verify installation
    uv run pytest

This installs:

* All OpenSTEF packages in editable mode
* Development tools (linting, testing, documentation)
* Pre-commit hooks for code quality

Package-Specific Development
----------------------------

To work on individual packages:

.. code-block:: bash

    # Install specific package in development mode
    cd packages/openstef-models
    uv pip install -e .

    # Or install with development dependencies
    uv sync --dev

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
    from openstef_models import forecasting
    from openstef_beam import evaluation

    # Not: from openstef.models import forecasting

**Memory Issues**

For large datasets, consider:

* Installing packages with specific memory optimizations
* Using data streaming approaches
* Configuring appropriate chunk sizes

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/OpenSTEF/openstef/issues>`_
2. Review the :doc:`../contribute/index` guide
3. Visit our :ref:`support` page for community resources
4. Contact us at short.term.energy.forecasts@alliander.com

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
* For Apple Silicon (M1/M2), ensure you're using compatible wheel distributions

Linux
-----

* Most distributions work out of the box
* For Ubuntu/Debian: ``sudo apt-get install python3-dev``
* For RHEL/CentOS: ``sudo yum install python3-devel``

Next Steps
==========

After installation:

1. Read the :doc:`quick_start` guide
2. Explore :doc:`tutorials` for hands-on examples
3. Check the :doc:`../api/index` for detailed documentation
4. Review :doc:`intro/index` to understand OpenSTEF's capabilities

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