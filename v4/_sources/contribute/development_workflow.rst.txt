.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

.. _development_workflow:

####################
Development workflow
####################

This guide walks you through the typical development workflow for contributing to OpenSTEF 4.0.

Workflow summary
================

To keep your work well organized, with readable history, and make it easier for project 
maintainers to see what you've done and why, we recommend the following:

* Don't make changes in your local ``main`` branch!
* Before starting a new set of changes, fetch all changes from ``upstream/main``
* Create a new *feature branch* for each feature or bug fix — "one task, one branch"
* Name your branch descriptively — e.g. ``feature/123-add-neural-network-model`` or ``bugfix/456-fix-data-validation``
* Use ``poe all`` to check your code before committing
* If you get stuck, reach out on the `LF Energy Slack workspace <https://slack.lfenergy.org/>`__ (#openstef channel)

Overview
--------

After :ref:`setting up a development environment <development_setup>`, the typical workflow is:

#. Fetch all changes from ``upstream/main``
#. Start a new *feature branch* from ``upstream/main``
#. Make your changes and test them with ``poe all --check``
#. Commit your changes and push to your fork
#. Open a pull request

Using Poe the Poet for development tasks
========================================

OpenSTEF 4.0 uses `Poe the Poet <https://poethepoet.natn.io/>`_ to manage development tasks. 
All common development operations are available through ``poe`` commands.

Essential commands
------------------

.. code-block:: bash

    # Get help on available tasks
    poe --help

    # Run all checks and fixes (linting, formatting, tests, etc.)
    poe all

    # Run all checks without making changes (CI mode)
    poe all --check

    # Individual tasks
    poe lint              # Lint code with ruff (and fix issues)
    poe lint --check      # Lint without fixing
    poe format            # Format code with ruff
    poe format --check    # Check formatting without changing files
    poe type              # Type check with pyright
    poe tests             # Run tests with pytest
    poe doctests          # Run docstring examples
    poe reuse             # Check license compliance
    poe reuse --fix       # Fix license headers automatically

The most important command is ``poe all --check``, which must pass before your PR can be merged.

.. _make-feature-branch:

Create a new feature branch
===========================

When you are ready to make changes, start by updating your local main branch and 
creating a new feature branch:

.. code-block:: bash

    # Make sure you're on main and it's up to date
    git checkout main
    git pull upstream main

    # Create and switch to a new feature branch
    git checkout -b feat/123-your-feature-description

Branch naming conventions
-------------------------

Use descriptive branch names that follow these patterns:

* ``feat/123-add-transformer-model`` - for new features (include issue number if available)
* ``fix/456-handle-missing-weather-data`` - for bug fixes  
* ``docs/789-improve-installation-guide`` - for documentation improvements
* ``refactor/cleanup-feature-engineering`` - for code refactoring
* ``perf/optimize-forecasting-pipeline`` - for performance improvements

The prefix should match the conventional commit type you'll use for the main changes.

The editing workflow
====================

1. **Make your changes** - Edit the code, add tests, update documentation
2. **Check your work** - Run ``poe all --check`` to verify everything passes
3. **Fix any issues** - Use ``poe all`` to automatically fix formatting and linting issues
4. **Add and commit your changes**:

   .. code-block:: bash

       git add .
       git commit -m "feat: add transformer-based forecasting model

       - Implement attention mechanism for temporal dependencies
       - Add tests with 95% coverage
       - Update documentation with usage examples
       
       Closes #123"

5. **Push to your fork**:

   .. code-block:: bash

       git push origin feature/123-your-feature-description

Commit message guidelines
-------------------------

OpenSTEF uses `Conventional Commits <https://www.conventionalcommits.org/>`_ for clear, 
structured commit messages that enable automated changelog generation and semantic versioning.

**Format:**

.. code-block:: text

    <type>[optional scope]: <description>

    [optional body]

    [optional footer(s)]

**Types:**

* ``feat``: A new feature for the user
* ``fix``: A bug fix  
* ``docs``: Documentation only changes
* ``style``: Changes that do not affect the meaning of the code (white-space, formatting, etc)
* ``refactor``: A code change that neither fixes a bug nor adds a feature
* ``perf``: A code change that improves performance
* ``test``: Adding missing tests or correcting existing tests
* ``build``: Changes that affect the build system or external dependencies
* ``ci``: Changes to our CI configuration files and scripts
* ``chore``: Other changes that don't modify src or test files
* ``revert``: Reverts a previous commit

**Examples:**

.. code-block:: bash

    # Feature with scope
    git commit -m "feat(models): add transformer-based forecasting model"
    
    # Bug fix with body and footer
    git commit -m "fix(validation): handle missing weather data gracefully
    
    Previously the validation would crash when weather data was missing.
    Now it logs a warning and continues with available features.
    
    Fixes #456"
    
    # Documentation update
    git commit -m "docs: update installation guide for uv workspace"
    
    # Breaking change
    git commit -m "feat!: change forecast output format to include uncertainty
    
    BREAKING CHANGE: forecast() now returns a DataFrame with columns 
    ['forecast', 'lower', 'upper'] instead of a Series"

**Guidelines:**

* Use the imperative mood ("Add feature" not "Added feature")
* Keep the first line under 50 characters
* Include a blank line before the body
* Reference issues with "Closes #123" or "Fixes #456"
* Explain *what* and *why*, not just *how*
* Use ``!`` after the type to indicate breaking changes

Testing your changes
====================

Before submitting your pull request, make sure all tests pass:

.. code-block:: bash

    # Run all tests
    poe tests

    # Run tests for specific markers
    poe tests --markers "not slow"

    # Run doctests
    poe doctests

    # Check test coverage
    poe tests
    poe report

Code quality checks
===================

OpenSTEF 4.0 enforces high code quality standards. Before submitting your PR:

.. code-block:: bash

    # Run all quality checks (must pass for PR merge)
    poe all --check

This command runs:

* **REUSE compliance** - Ensures all files have proper license headers
* **Linting** - Checks code style and potential issues with ruff
* **Formatting** - Verifies code formatting with ruff
* **Type checking** - Validates type hints with pyright  
* **Tests** - Runs the full test suite
* **Doctests** - Verifies all code examples in docstrings work

For detailed information about coding standards and conventions, see our 
:doc:`code_style_guide`.

If any checks fail, you can fix them automatically:

.. code-block:: bash

    # Fix most issues automatically
    poe all

License compliance
==================

OpenSTEF uses the `REUSE <https://reuse.software/>`_ specification for license compliance. 
All files must have proper license headers.

.. code-block:: bash

    # Check license compliance
    poe reuse

    # Automatically add missing license headers
    poe reuse --fix

The ``reuse --fix`` command automatically adds the correct license header to new files:

.. code-block:: python

    # SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
    #
    # SPDX-License-Identifier: MPL-2.0

Opening a pull request
======================

When you're ready to submit your changes:

1. **Push your branch** to your fork on GitHub
2. **Open a pull request** from your branch to ``main``
3. **Fill out the PR template** with a clear description
4. **Link to the relevant issue** (e.g., "Closes #123")
5. **Request review** from appropriate maintainers

PR requirements
---------------

Before your PR can be merged:

* ``poe all --check`` must pass
* All tests must pass  
* Code coverage should not decrease significantly
* Documentation must be updated for new features
* At least one maintainer must approve the changes

Example PR description:

.. code-block:: markdown

    ## Summary
    
    This PR implements a transformer-based neural network model for time series forecasting.
    
    ## Changes
    
    - Add `TransformerForecaster` class in `openstef_models.models.forecasting`
    - Implement attention mechanism for temporal dependencies
    - Add tests with 95% coverage
    - Update documentation with usage examples
    
    ## Testing
    
    - [x] All existing tests pass
    - [x] New tests added with good coverage
    - [x] Doctests pass
    - [x] Manual testing on sample datasets
    
    Closes #123

Updating your pull request
==========================

If you need to make changes after opening your PR:

1. **Make the changes** in your local branch
2. **Run quality checks**: ``poe all --check``  
3. **Commit the changes**
4. **Push to your branch**: ``git push origin your-branch-name``

The PR will automatically update with your new commits.

Working with the monorepo
=========================

OpenSTEF 4.0 uses a monorepo structure with multiple packages. When making changes:

* **Changes in** ``openstef-models`` **affect the core forecasting functionality**
* **Changes in** ``openstef-beam`` **affect evaluation and analysis tools**
* **Changes in** ``docs`` **affect documentation**
* **Changes in** ``openstef-core`` **affect shared dataset types and utilities and therefore can impact both `openstef-models` and `openstef-beam`**

If your changes span multiple packages, make sure to:

* Test all affected packages
* Update documentation in all relevant places
* Consider backward compatibility between packages

.. include:: _getting_help.rst
