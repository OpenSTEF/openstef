.. comment:
    SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

.. _user_guides:

User guides
===========

This page contains guides and links to resources that show how OpenSTEF can be used.

Pipelines - high level functionality
------------------------------------
Pipelines (see :ref:`concepts <concepts>` for definition) offer an easy way to use OpenSTEF for training models, generating forecasts, and evaluation of forecasting performance. 

The following pipelines are available:

- :mod:`openstef.pipeline.train_model`
- :mod:`openstef.pipeline.create_forecast`
- :mod:`openstef.pipeline.optimize_hyperparameters`
- :mod:`openstef.pipeline.create_component_forecast`
- :mod:`openstef.pipeline.create_basecase_forecast`
- :mod:`openstef.pipeline.train_create_forecast_backtest`

The easiest way to get started and get familiar with pipelines is to have a look at 
`this GitHub repository that contains an assortment of examples Jupyter notebooks <https://github.com/OpenSTEF/openstef-offline-example>`_, including example data.
Each of these example notebooks can be ran locally without any setup required, apart from the `installation of the OpenSTEF package <https://pypi.org/project/openstef/>`_. 

- Usage of the train model pipeline is demonstrated in 
  `this example Jupyter Notebook <https://github.com/OpenSTEF/openstef-offline-example/blob/master/examples/01.%20Train%20a%20model%20using%20high-level%20pipelines.ipynb>`_.
- Usage of the create forecast pipeline is demonstrated in 
  `this example Jupyter Notebook <https://github.com/OpenSTEF/openstef-offline-example/blob/master/examples/04.%20Test_on_difficult_cases.ipynb>`_.
- Usage of the train model and forecast backtest pipeline is demonstrated in multiple notebooks, for instance
  `this example Jupyter Notebook <https://github.com/OpenSTEF/openstef-offline-example/blob/master/examples/02.%20Evaluate%20performance%20using%20Backtest%20Pipeline.ipynb>`_.

The notebooks mentioned above are aimed towards a backtesting setting.
More in depth information on how to use and implement the pipelines in an operational setting, including code examples, is provided on the following page:

.. toctree::
    :maxdepth: 1

    pipelines_user_guide


Forecasting application - full implementation
---------------------------------------------

For those who wish to set up a fully functioning forecasting application that is ready to be used in an operational setting, a 
`GitHub repository with a reference implementation <https://github.com/OpenSTEF/openstef-reference>`_  is available. 
The example implementation includes databases, a user interface, and example data.
More information on what the architecture of such an application could look like can be found :ref:`here <application-architecture>`.

.. include:: dashboard.rst
Screenshot of the operational dashboard showing the key functionality of OpenSTEF. Dashboard documentation can be found `here <https://raw.githack.com/OpenSTEF/.github/main/profile/html/openstef_dashboard_doc.html>`_.

Example Jupyter notebooks
-------------------------
Jupyter Notebooks demonstrating some of OpenSTEF's main functionality can be found at: https://github.com/OpenSTEF/openstef-offline-example.