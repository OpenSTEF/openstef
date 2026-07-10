.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
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

    .. grid-item-card:: :fa:`chart-line` BEAM Package
        :link: beam-api
        :link-type: ref

        BEAM (Backtesting, Evaluation, Analysis, and Metrics) for forecasting models

    .. grid-item-card:: :fa:`layer-group` Meta Package
        :link: meta-api
        :link-type: ref

        Ensemble forecasting and preset workflows

    .. grid-item-card:: :fa:`wand-sparkles` Foundation Models Package
        :link: foundation-models-api
        :link-type: ref

        Pretrained foundation-model forecasters on an ONNX runtime

.. toctree::
   :maxdepth: 2
   :hidden:

   core
   models
   beam
   meta
   foundation_models

Core Package (:mod:`openstef_core`)
-----------------------------------

.. currentmodule:: openstef_core

.. autosummary::

   datasets
   utils
   base_model
   exceptions
   mixins
   testing
   transforms
   types

Models Package (:mod:`openstef_models`)
---------------------------------------

.. currentmodule:: openstef_models

.. autosummary::

   models
   workflows
   presets
   explainability
   mixins
   integrations
   transforms
   utils

BEAM Package (:mod:`openstef_beam`)
-----------------------------------

.. currentmodule:: openstef_beam

.. autosummary::

   metrics
   backtesting
   analysis
   evaluation
   benchmarking

Meta Package (:mod:`openstef_meta`)
-----------------------------------

.. currentmodule:: openstef_meta

.. autosummary::

   models
   presets
   utils

Foundation Models Package (:mod:`openstef_foundation_models`)
-------------------------------------------------------------

.. currentmodule:: openstef_foundation_models

.. autosummary::

   models
   presets
   inference
   integrations
