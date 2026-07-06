.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. title:: OpenSTEF documentation

.. module:: openstef

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   User Guide <user_guide/index>
   Examples <examples>
   API <api/index>
   Community <project/index>
   Contributing <contribute/index>

##################################
OpenSTEF 4.0.0 documentation
##################################


OpenSTEF is a library for creating short term forecasts for the energy sector. 
It contains all components for the machine learning pipeline required to make a forecast.

Install
=======

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

    .. tab-item:: other

        .. rst-class:: section-toc

        :doc:`user_guide/getting_started/installation`


For more detailed instructions, see the
:doc:`installation guide <user_guide/getting_started/installation>`.

Learn
=====

.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2
        :columns: 6

        **How to use OpenSTEF?**
        ^^^

        * :doc:`user_guide/getting_started/installation`
        * :doc:`user_guide/getting_started/quick_start_tutorial`
        * :doc:`user_guide/guides/forecasting`
        * :doc:`examples`

    .. grid-item-card::
        :padding: 2
        :columns: 6

        **What can OpenSTEF do?**
        ^^^

        * :doc:`user_guide/concepts/intro_to_energy_forecasting`
        * :doc:`user_guide/concepts/models`
        * :doc:`user_guide/guides/probabilistic_forecasting`
        * :doc:`user_guide/concepts/beam`
        * :doc:`user_guide/concepts/component_splitting`


    .. grid-item-card::
        :padding: 2
        :columns: 12

        **Reference**
        ^^^

        .. grid:: 1 1 2 2
            :class-row: sd-align-minor-center

            .. grid-item::

                * :doc:`API reference <api/index>`
                * :doc:`Concepts <user_guide/concepts/index>`

            .. grid-item::

                OpenSTEF packages:

                - :doc:`openstef-models <api/models>` forecasting models and feature engineering
                - :doc:`openstef-beam <api/beam>` backtesting, evaluation, analysis and metrics
                - :doc:`openstef-core <api/core>` utilities, dataset types, shared types
                - :doc:`openstef-meta <api/meta>` metalearning and model selection
                - :doc:`openstef-foundation-models <api/foundation_models>` pretrained foundation-model forecasters on ONNX


What's new
==========

.. grid:: 1 1 2 2

    .. grid-item::

       Learn about new features and API changes.

    .. grid-item::

        * `Release notes on GitHub <https://github.com/OpenSTEF/openstef/releases>`_

Contribute
==========

.. grid:: 1 1 2 2
    :class-row: sd-align-minor-center

    .. grid-item::

        OpenSTEF is an LF Energy community maintained for and by its users. See
        :ref:`contributing` for the many ways you can help!

    .. grid-item::
        .. rst-class:: section-toc

        * :ref:`submitting-a-bug-report`
        * :ref:`contribute_guide`
        * :ref:`development_workflow`
        * :ref:`contribution_guideline`



About OpenSTEF
==============

.. grid:: 1 1 2 2
    :class-row: sd-align-minor-center

    .. grid-item::

        Here you can find all information about the OpenSTEF community.

    .. grid-item::
        .. rst-class:: section-toc

        * `Linux Foundation project page <https://www.lfenergy.org/projects/openstef/>`_
        * `Video About OpenSTEF <https://www.lfenergy.org/forecasting-to-create-a-more-resilient-optimized-grid/>`_
        * :doc:`project/committee`
        * :doc:`project/maintainers`
        * :doc:`project/citing`
        * :doc:`project/license`
