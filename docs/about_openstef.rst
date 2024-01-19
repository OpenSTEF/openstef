.. comment:
    SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

About OpenSTEF
==============

The energy transition poses new challenges to all parties within the energy sector. Grid operators, grappling with the upsurge in renewable energy and heightened electrification, find their grid capacities nearing physical limitations. Therefore, it is imperative to forecast grid load in the upcoming hours to days, enabling the anticipation of local congestion and thereby optimal utilization of existing assets.

OpenSTEF provides a complete software stack specifically engineered to forecast the load on the electricity grid for the next hours to days. Given a timeseries of measured (net) load or generation, a fully automated machine learning pipeline is executed which delivers a probabilistic forecast of future load. This is applicable to energy consumption, renewable generation, or a combination of the two. OpenSTEF does not stop at forecating: it validates input data, combines measurements with external predictors such as weather data and market prices, trains any scikit-learn compatible machine learning model, and delivers the forecast via both an API and an (expert) graphical user interface. 

The entire stack, crafted on open-source technology and adhering to standards, is organized in a microservice architecture optimized for cloud-deployment.

Features:
---------
.. image:: _static/infinity.png
  :width: 50

* **Resilient**: As forecast availability is critical in energy sector applications OpenSTEF deploys multiple fallback stategies. This way a forecast is always available. When a fallback forecast is issued this is always labeled as such making it possible to reconstruct on which forecasts a decision is based.

.. image:: _static/crane.png
  :width: 50

* **Cloud based and platform agnostic** OpenSTEF is fully containerized and runs on any container platform. A reference implementation is available that can be deployed directly. Most users will however have a unique IT landscape in which case the modular nature of OpenSTEF enables users to easily addapt openSTEF to their environment.
.. image:: _static/probability.png
  :width: 50

* **Probabilistic forecasts** Making decisions can be difficult, openSTEF enables to make risk-based decisions by providing probabilistic forecasts. This way users can work towards a standard policy to react to predicted events.
.. image:: _static/power-source.png
  :width: 50

* **Split forecasts in energy source components** With an increasing fraction of renewable sources in the energy mix balancing the grid can be challenging. OpenSTEF provides insight in the fraction of wind and solar power generation. This is relevant for `EU commision regulation No. 543/2013 <https://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri=OJ:L:2013:163:0001:0012:EN:PDF>`_

Repositories:
---------
There are four repositories regarding OpenSTEF:

* OpenSTEF: Basis of all the repositories. Automatic machine learning pipelines. Builds the Opensource Short Term Forecasting package.

* OpenSTEF-dbc: Provides (company specific) database connector for OpenSTEF package.

* OpenSTEF-reference: Deploy the entire OpenSTEF stack on your machine. Provides a reference implementation of the OpenSTEF stack including datamodels, databases and UI.

* OpenSTEF-offline-example: Provides Jupyter Notebooks showing how to use OpenSTEF and apply it's functionality to your usecase.

These repositories can be found on the Github page: https://github.com/OpenSTEF/.
