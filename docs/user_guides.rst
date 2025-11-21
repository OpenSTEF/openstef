.. comment:
    SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <openstef@lfenergy.org>
    SPDX-License-Identifier: MPL-2.0

.. _user_guides:

User guides
===========

This page contains guides and links to resources that show how OpenSTEF can be used.

Pipelines - high level functionality
------------------------------------
OpenSTEF is designed around Pipelines (see :ref:`concepts <concepts>` for definition). Pipelines offer an easy way to train models, generate forecasts, and evaluate  forecasting performance.


The following pipelines are available:

- :mod:`openstef.pipeline.train_model`
- :mod:`openstef.pipeline.create_forecast`
- :mod:`openstef.pipeline.optimize_hyperparameters`
- :mod:`openstef.pipeline.create_component_forecast`
- :mod:`openstef.pipeline.create_basecase_forecast`
- :mod:`openstef.pipeline.train_create_forecast_backtest`

A great way to get started and become familiar with OpenSTEF pipelines is to have a look at
`this GitHub repository that contains an assortment of Jupyter notebook examples <https://github.com/OpenSTEF/openstef-offline-example>`_. The repository
even includes example data.

You can run each example notebook locally without any setup required, apart from the `installation of the OpenSTEF package <https://pypi.org/project/openstef/>`_.

We encourage you to check out all the examples, but here is a list to get you started:

- `How to train a model <https://github.com/OpenSTEF/openstef-offline-example/blob/master/examples/01.%20Train%20a%20model%20using%20high-level%20pipelines.ipynb>`_.
- `How to create a forecast <https://github.com/OpenSTEF/openstef-offline-example/blob/master/examples/04.%20Test_on_difficult_cases.ipynb>`_.
- `How evaluate the performance of model using a backtest  <https://github.com/OpenSTEF/openstef-offline-example/blob/master/examples/02.%20Evaluate%20performance%20using%20Backtest%20Pipeline.ipynb>`_.

For more in-depth information on how to use and implement the pipelines in an operational setting, including code examples, see the :ref:`pipeline_user_guide` section of this documentation.


Deploy as a full Forecasting application
----------------------------------------
If you would like to setup a full forecasting application that is ready to be used in an operational setting with a
backend datastore and graphical user interface frontent, this
`GitHub repository contains a reference implementation <https://github.com/OpenSTEF/openstef-reference>`_  you can use as a starting point.
This example implementation includes databases, a user interface, and example data.

More information on what the architecture of such an application could look like can be found :ref:`here <application-architecture>`.

.. include:: dashboard.rst
Screenshot of the operational dashboard showing the key functionality of OpenSTEF. Dashboard documentation can be found `here <https://raw.githack.com/OpenSTEF/.github/main/profile/html/openstef_dashboard_doc.html>`_.

Example Jupyter notebooks
-------------------------
Jupyter Notebooks demonstrating some of OpenSTEF's main functionality can be found at: https://github.com/OpenSTEF/openstef-offline-example.
