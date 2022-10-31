.. comment:
    SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

.. _user_guides:

User guides
===========

This page contains guides and links to resources that show how OpenSTEF can be used.

Using pipelines
---------------
Pipelines (see :ref:`concepts <concepts>` for definition) offer an easy way to use OpenSTEF for training models, generating forecasts, and evaluation of forecasting performance. 
Each of the user guides below discusses the functionality and purpose of a pipeline.

.. toctree::
    :maxdepth: 1

    pipeline_user_guides/train_model_pipeline
    pipeline_user_guides/create_forecast_pipeline
    pipeline_user_guides/optimize_hyperparameters_pipeline
    pipeline_user_guides/create_components_forecast_pipeline
    pipeline_user_guides/create_base_case_forecast_pipeline
    pipeline_user_guides/train_model_and_forecast_backtest


Forecasting application - full implementation
---------------------------------------------

For those who wish to set up a fully functioning forecasting application, a 
`GitHub repository with a reference implementation <https://github.com/OpenSTEF/openstef-reference>`_  is available. 
The example implementation includes a databases, user interface, and example data.
More information on what the architecture of such an application could look like can be found :ref:`here <application-architecture>`.

.. include:: dashboard.rst
Screenshot of the operational dashboard showing the key functionality of OpenSTEF. Dashboard documentation can be found `here <https://raw.githack.com/OpenSTEF/.github/main/profile/html/openstef_dashboard_doc.html>`_.

Example Jupyter notebooks
-------------------------
Jupyter Notebooks demonstrating some of OpenSTEF's main functionality can be found at: https://github.com/OpenSTEF/openstef-offline-example.