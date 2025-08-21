.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

API Reference
=============

This is the complete API reference for OpenSTEF. The API is organized into several packages:

.. grid:: 1 1 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: :fa:`cube` Core Package
        :link: core-api
        :link-type: ref
        
        Core data structures, datasets, and utilities

    .. grid-item-card:: :fa:`robot` Models Package
        :link: models-api
        :link-type: ref
        
        Machine learning models and feature engineering

.. _core-api:

Core Package (:mod:`openstef_core`)
-----------------------------------

.. currentmodule:: openstef_core

**Core Modules:**

.. autosummary::
   :toctree: generated/
   :template: package_overview.rst

   datasets
   utils
   base_model
   exceptions

.. _models-api:

Models Package (:mod:`openstef_models`)
---------------------------------------

.. currentmodule:: openstef_models

**Model Modules:**

.. autosummary::
   :toctree: generated/
   :template: package_overview.rst

   feature_engineering
   models
   pipelines
   postprocessing
   explainability
   exceptions


