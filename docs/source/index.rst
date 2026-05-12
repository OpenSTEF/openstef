OpenSTEF Documentation
======================

**Open Short-Term Energy Forecasting** is a Python library for building accurate short-term load forecasts in the power grid domain. It provides complete machine learning pipelines—from data preprocessing and feature engineering through model training to probabilistic forecasting—with built-in domain knowledge specific to energy systems.

OpenSTEF predicts electrical load hours to days ahead, generating forecasts with uncertainty bandwidths rather than single-point predictions. This makes it suitable for operational decisions like congestion management, transport capacity planning, and grid loss prediction.

Who is it for?
--------------

OpenSTEF is designed for data scientists and engineers working on energy forecasting problems at utilities, grid operators, or research institutions. You should be comfortable with Python and have access to historical load measurements and weather forecast data. No deep expertise in energy systems is required—the library encodes that domain knowledge for you.

Where to start
--------------

If you're new to OpenSTEF, begin with the installation guide to get the library set up, then work through the quickstart tutorial which walks you through loading data, training a model, and generating your first forecast. The user guide covers core concepts like feature engineering, model selection, and forecast evaluation in depth. For production deployments, the deployment section provides patterns for scheduling and orchestrating forecast pipelines. The API reference documents every module and function.

Community and support
---------------------

OpenSTEF is an `LF Energy <https://www.lfenergy.org/>`_ project, developed in the open and welcoming contributions of all kinds.

- **Slack:** Join the conversation at https://slack.lfenergy.org/
- **Email:** openstef@lfenergy.org
- **Community meetings:** `Four-weekly open call <https://lf-energy.atlassian.net/wiki/spaces/OS/pages/32278358/OpenSTEF+four-weekly+community+meeting>`_
- **GitHub:** https://github.com/OpenSTEF/openstef — bug reports and feature requests go here

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :hidden:

   getting_started/index
   user_guide/index
   concepts/index
   architecture/index
   faq
   changelog
   contribute/index
   api/index
