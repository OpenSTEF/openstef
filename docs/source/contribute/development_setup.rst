.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

.. _development_setup:

=====================================
Setting up OpenSTEF for development
=====================================

This guide will help you set up a development environment for OpenSTEF 4.0, 
which uses a modern Python development stack with uv and a monorepo workspace structure.

Prerequisites
=============

Python version
--------------

OpenSTEF 4.0 requires Python 3.12 or higher. You can check your Python version with:

.. code-block:: bash

    python --version

Installing uv
=============

OpenSTEF 4.0 uses `uv <https://docs.astral.sh/uv/>`_ as the primary dependency manager and 
Python project manager. uv is a fast, Rust-based Python package manager that handles both 
dependency resolution and virtual environments.

Install uv using the recommended method for your platform:

.. tab-set::

    .. tab-item:: macOS and Linux

        .. code-block:: bash

            curl -LsSf https://astral.sh/uv/install.sh | sh

    .. tab-item:: Windows

        .. code-block:: powershell

            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    .. tab-item:: pipx

        .. code-block:: bash

            pipx install uv

    .. tab-item:: pip

        .. code-block:: bash

            pip install uv

For more installation options, see the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

Cloning the repository
======================

Clone the OpenSTEF repository and navigate to the project directory:

.. code-block:: bash

    git clone https://github.com/OpenSTEF/openstef.git
    cd openstef

Setting up the development environment
======================================

OpenSTEF 4.0 uses a workspace structure with multiple packages:

* ``openstef-models``: Core forecasting models, feature engineering, and data processing
* ``openstef-beam``: Backtesting, evaluation, analysis and metrics  
* ``docs``: Documentation source
* ``openstef-compatibility``: Compatibility layer for OpenSTEF 3.x (coming soon)
* ``openstef-foundational-models``: Deep learning and foundational models (coming soon)

Install the development dependencies using uv:

.. code-block:: bash

    uv sync --all-groups --all-extras

This will:

1. Create a virtual environment (if one doesn't exist)
2. Install all workspace packages in development mode
3. Install all development dependencies including testing, linting, and documentation tools
4. Install all optional extras for comprehensive development

Understanding the workspace structure
=====================================

OpenSTEF 4.0 uses a monorepo workspace structure defined in ``pyproject.toml``:

.. code-block:: toml

    [tool.uv.workspace]
    members = [
      "packages/openstef-models",
      "packages/openstef-beam", 
      "docs",
    ]

This means:

* All packages are developed together in a single repository
* Dependencies between packages are automatically resolved
* You can work on multiple packages simultaneously
* Changes in one package are immediately available to other packages

Verifying your installation
===========================

Test that your environment is set up correctly:

.. code-block:: bash

    # Check that uv is working
    uv --version

    # Check that you can run the task runner
    uv run poe --help

    # Run a quick test to verify everything works
    uv run poe all --check

If everything is working correctly, the last command should run all linting, formatting, 
type checking, and tests without errors.

Next steps
==========

Now that you have a working development environment, you can:

1. Read the :doc:`development_workflow` guide to understand how to contribute
2. Explore the :doc:`document` guide if you want to contribute to documentation
3. Look at the `good first issues <https://github.com/OpenSTEF/openstef/labels/good%20first%20issue>`_ 
   to find something to work on

Development tools overview
==========================

Note that also OpenSTEF 4.0 uses several modern development tools:

* **uv**: Package manager and virtual environment handler (`uv website <https://docs.astral.sh/uv/>`_)
* **Poe the Poet**: Task runner for common development tasks (`poethepoet website <https://poethepoet.natn.io/>`_)
* **Ruff**: Lightning-fast linting and formatting (`ruff website <https://docs.astral.sh/ruff/>`_)
* **Pyright**: Type checking (`pyright website <https://microsoft.github.io/pyright/>`_)
* **pytest**: Testing framework with coverage reporting (`pytest website <https://pytest.org/>`_)
* **Sphinx**: Documentation generation (`sphinx website <https://www.sphinx-doc.org/>`_)
* **REUSE**: License compliance checking (`reuse website <https://reuse.software/>`_)

All these tools are configured and ready to use through the ``poe`` task runner. 
See ``poe --help`` for available commands.

Building documentation
======================

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
    development environment. Make sure you've run ``uv sync --all-groups --all-extras`` first.
