.. comment:
    SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

About OpenSTEF
==============

The energy transition poses new challenges to all parties in the energy sector. 
For grid operators, the rise in renewable energy and electrification of energy consumption leads to the capacity of the grid to near its fysical constraints.
Forecasting the load on the grid in the next hours to days is essential for anticipating on local congestion and making the most of existing assets. 

OpenSTEF provides a complete software stack which forecasts the load on the electricity grid for the next hours to days. 
Given a timeseries of measured (net) load or generation, a fully automated machine learning pipeline is executed which delivers a probabilistic forecast of future load. 
is works for energy consumption, (renewable) generation or a combination of both. OpenSTEF performs validation on the input data, combines measurements with external predictors such as weather data and market prices, trains any scikit-learn compatible machine learning model, and delivers the forecast via both an API and an (expert) graphical user interface. 
The stack is based on open source technology and standards and is organized in a microservice architecture optimized for cloud-deployment.

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
