.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _concepts:

========
Concepts
========

Understand the domain and library architecture behind OpenSTEF.

.. grid:: 1 1 2 2
    :gutter: 4
    :padding: 2 2 0 0

    .. grid-item-card:: :fa:`bolt` Intro to Energy Forecasting
        :link: intro_to_energy_forecasting
        :link-type: doc

        How short-term energy forecasting works and why it matters for grid operations.

    .. grid-item-card:: :fa:`brain` Models
        :link: models
        :link-type: doc

        The machine learning models available in OpenSTEF and when to use each.

    .. grid-item-card:: :fa:`wand-magic-sparkles` Metalearning
        :link: metalearning
        :link-type: doc

        Automatic model selection and hyperparameter optimization across targets.

    .. grid-item-card:: :fa:`chart-line` BEAM
        :link: beam
        :link-type: doc

        Backtesting, Evaluation, Analysis, and Metrics — the BEAM framework.

    .. grid-item-card:: :fa:`puzzle-piece` Component Splitting
        :link: component_splitting
        :link-type: doc

        Decompose aggregated load into individual energy components.

    .. grid-item-card:: :fa:`wand-sparkles` Foundation Models
        :link: foundation_models
        :link-type: doc

        Zero-shot pretrained forecasting on an ONNX runtime, no per-target training.

.. toctree::
    :maxdepth: 1
    :hidden:

    Intro to Energy Forecasting <intro_to_energy_forecasting>
    models
    metalearning
    beam
    component_splitting
    foundation_models
